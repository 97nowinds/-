/* Bound the entire read (headers and body), so one stuck request cannot stop polling. */
(() => {
  async function fetchJSON(url, {timeoutMs = 5000} = {}) {
    const controller = new AbortController();
    let timer;
    const deadline = new Promise((_, reject) => {
      timer = setTimeout(() => {
        controller.abort();
        reject(new Error('数据读取超时'));
      }, timeoutMs);
    });
    try {
      return await Promise.race([
        (async () => {
          const response = await fetch(url, {cache:'no-store', signal:controller.signal});
          if (!response.ok) throw new Error(`数据接口返回 ${response.status}`);
          return await response.json();
        })(),
        deadline,
      ]);
    } finally {
      clearTimeout(timer);
    }
  }
  window.LivePolling = {fetchJSON};
})();
