/* tslint:disable */
/* eslint-disable */

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly main: (a: number, b: number) => number;
    readonly wasm_bindgen_5c4e04957574c935___closure__destroy___dyn_core_1913cbbf38a9a7c___ops__function__FnMut__web_sys_8f3db83bf8ea1184___features__gen_InputEvent__InputEvent____Output_______: (a: number, b: number) => void;
    readonly wasm_bindgen_5c4e04957574c935___closure__destroy___dyn_core_1913cbbf38a9a7c___ops__function__FnMut__wasm_bindgen_5c4e04957574c935___JsValue____Output_______: (a: number, b: number) => void;
    readonly wasm_bindgen_5c4e04957574c935___closure__destroy___dyn_core_1913cbbf38a9a7c___ops__function__FnMut__js_sys_8ad58fd14919b6b7___Array____Output_______: (a: number, b: number) => void;
    readonly wasm_bindgen_5c4e04957574c935___convert__closures_____invoke___js_sys_8ad58fd14919b6b7___Array__web_sys_8f3db83bf8ea1184___features__gen_ResizeObserver__ResizeObserver_____: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_5c4e04957574c935___convert__closures_____invoke___web_sys_8f3db83bf8ea1184___features__gen_InputEvent__InputEvent_____: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_5c4e04957574c935___convert__closures_____invoke___wasm_bindgen_5c4e04957574c935___JsValue_____: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_5c4e04957574c935___convert__closures_____invoke___js_sys_8ad58fd14919b6b7___Array_____: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_5c4e04957574c935___convert__closures_____invoke______: (a: number, b: number) => void;
    readonly __wbindgen_malloc_command_export: (a: number, b: number) => number;
    readonly __wbindgen_realloc_command_export: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_alloc_command_export: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_exn_store_command_export: (a: number) => void;
    readonly __wbindgen_free_command_export: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
