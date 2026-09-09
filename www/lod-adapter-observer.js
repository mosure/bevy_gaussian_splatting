// Observe the promises returned to the renderer. This module never requests an
// adapter/device of its own and never changes the renderer's request options.
export function observeRendererAdapter(gpu, wasmSha256, emit) {
  const requestAdapter = gpu.requestAdapter;
  let nextRequest = 0;
  let nextDevice = 0;
  const observedAdapters = new WeakSet();
  const errorText = error => String(error?.message ?? error).slice(0, 2048);
  const limits = device => Object.fromEntries([
    'maxBufferSize', 'maxStorageBufferBindingSize', 'maxStorageBuffersPerShaderStage',
  ].map(name => [name, device.limits[name]]));
  Object.defineProperty(gpu, 'requestAdapter', {configurable: true, value: function (...args) {
    const requestId = ++nextRequest;
    const options = args[0] ?? {};
    const record = {
      kind: 'browser_adapter', request_id: requestId, wasm_sha256: wasmSha256,
      source: 'renderer_request_adapter_promise', status: 'requested',
      options: {power_preference: options.powerPreference ?? null,
        force_fallback_adapter: options.forceFallbackAdapter ?? false,
        feature_level: options.featureLevel ?? null},
      info: null, error: null,
    };
    let promise;
    try { promise = Reflect.apply(requestAdapter, this, args); }
    catch (error) { emit({...record, status: 'rejected', error: errorText(error)}); throw error; }
    return promise.then(adapter => {
      if (adapter === null) { emit({...record, status: 'null'}); return adapter; }
      const info = adapter.info;
      record.info = {
        vendor: info?.vendor ?? '', architecture: info?.architecture ?? '',
        device: info?.device ?? '', description: info?.description ?? '',
        is_fallback_adapter: info?.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null,
      };
      emit({...record, status: 'returned'});
      if (observedAdapters.has(adapter)) throw new Error('Renderer received the same adapter from multiple requests; qualification is ambiguous');
      observedAdapters.add(adapter);
      const requestDevice = adapter.requestDevice;
      Object.defineProperty(adapter, 'requestDevice', {configurable: true, value: function (...deviceArgs) {
        const descriptor = deviceArgs[0] ?? {};
        const deviceRecord = {
          kind: 'browser_device', request_id: requestId, device_id: ++nextDevice,
          wasm_sha256: wasmSha256, source: 'renderer_request_device_promise',
          required_features: Array.from(descriptor.requiredFeatures ?? []),
          required_limits: {...(descriptor.requiredLimits ?? {})},
          status: 'requested', limits: null, error: null,
        };
        let devicePromise;
        try { devicePromise = Reflect.apply(requestDevice, this, deviceArgs); }
        catch (error) { emit({...deviceRecord, status: 'rejected', error: errorText(error)}); throw error; }
        return devicePromise.then(device => {
          emit({...deviceRecord, status: 'returned', limits: limits(device)});
          return device;
        }, error => { emit({...deviceRecord, status: 'rejected', error: errorText(error)}); throw error; });
      }});
      return adapter;
    }, error => { emit({...record, status: 'rejected', error: errorText(error)}); throw error; });
  }});
}
