/**
 * Service worker: the app works fully offline after the first visit, and from the second
 * visit on the page is cross-origin isolated (so onnxruntime can use several CPU threads).
 *
 * - Precaches the small app shell (asset_manifest.json entries marked `precache`: page,
 *   scripts, styles, icons). The models and the onnxruntime build are cached the first time
 *   the processing worker fetches them (main.js waits for this worker to take control before
 *   starting it), so they are downloaded exactly once, with real progress, and a visitor only
 *   gets the one runtime build their browser uses.
 * - vendor/ files live in their own unversioned cache: their paths carry the ORT version and
 *   never change, so an app update does not re-download 14-27 MB of WebAssembly.
 * - Every response it serves gets COOP/COEP/CORP headers. Static hosts like GitHub Pages
 *   cannot set headers, so this is the only way to get `crossOriginIsolated`. There is no
 *   forced reload: the first visit runs single-threaded, later visits get threads.
 * - Same-origin GET only. It never contacts any other origin (the page's CSP forbids it too).
 *
 * ASSET_MANIFEST_VERSION is stamped by `python -m tests.app.sync_asset_manifest`; a new value
 * is what makes browsers install the new version.
 */

const ASSET_MANIFEST_VERSION = "e764f7bcd5bddb17";
const APP_CACHE_NAME = `glare-off-app-${ASSET_MANIFEST_VERSION}`;
const VENDOR_CACHE_NAME = "glare-off-vendor";
const ISOLATION_HEADERS = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};

const scopeUrl = new URL(self.registration.scope);

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const manifestResponse = await fetch(new URL("asset_manifest.json", scopeUrl), { cache: "no-cache" });
      const manifest = await manifestResponse.json();
      const precacheUrls = manifest.assets.filter((asset) => asset.precache).map((asset) => new URL(asset.path, scopeUrl).href);
      const appCache = await caches.open(APP_CACHE_NAME);
      await appCache.addAll([scopeUrl.href, ...precacheUrls].map((url) => new Request(url, { cache: "no-cache" })));
      await appCache.put(new URL("asset_manifest.json", scopeUrl).href, new Response(JSON.stringify(manifest), { headers: { "Content-Type": "application/json" } }));
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      for (const cacheName of await caches.keys()) {
        if (cacheName.startsWith("glare-off-app-") && cacheName !== APP_CACHE_NAME) await caches.delete(cacheName);
      }
      await pruneVendorCache();
      await self.clients.claim();
    })(),
  );
});

/** Drop vendor files the current manifest no longer lists (after an ORT upgrade). */
async function pruneVendorCache() {
  const appCache = await caches.open(APP_CACHE_NAME);
  const manifestResponse = await appCache.match(new URL("asset_manifest.json", scopeUrl).href);
  if (!manifestResponse) return;
  const listedUrls = new Set((await manifestResponse.json()).assets.map((asset) => new URL(asset.path, scopeUrl).href));
  const vendorCache = await caches.open(VENDOR_CACHE_NAME);
  for (const request of await vendorCache.keys()) {
    if (!listedUrls.has(request.url)) await vendorCache.delete(request);
  }
}

function withIsolationHeaders(response) {
  if (!response || response.type === "opaque" || response.status === 0) return response;
  const headers = new Headers(response.headers);
  for (const [name, value] of Object.entries(ISOLATION_HEADERS)) headers.set(name, value);
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

/** Cache-first. A miss is fetched, returned at once (streaming), and cached in the background. */
async function respond(event) {
  const request = event.request;
  const url = new URL(request.url);
  const isNavigation = request.mode === "navigate";
  const isVendorFile = url.pathname.startsWith(new URL("vendor/", scopeUrl).pathname);
  const cache = await caches.open(isVendorFile ? VENDOR_CACHE_NAME : APP_CACHE_NAME);
  const cacheKey = isNavigation && url.pathname === scopeUrl.pathname ? scopeUrl.href : url.href.split("#")[0];
  const cached = await cache.match(cacheKey, { ignoreSearch: isNavigation });
  if (cached) return withIsolationHeaders(cached);
  const networkResponse = await fetch(request);
  if (networkResponse.ok && networkResponse.type === "basic") {
    event.waitUntil(cache.put(cacheKey, networkResponse.clone()));
  }
  return withIsolationHeaders(networkResponse);
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== scopeUrl.origin || !url.pathname.startsWith(scopeUrl.pathname)) return;
  event.respondWith(respond(event));
});
