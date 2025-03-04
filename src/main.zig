const std = @import("std");
const c = @import("c.zig");

var device: c.CUdevice = undefined;
var context: c.CUcontext = undefined;
var module: c.CUmodule = undefined;
var function: c.CUfunction = undefined;
var totalMem: usize = undefined;

const matSum = @embedFile("matSum.ptx");
const vecAdd = @embedFile("VecAdd.ptx");

const N = 1024;

inline fn GiB(bytes: usize) f32 {
    return @as(f32, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

fn init() void {
    checkCuda(c.cuInit(0), @src());
    var device_count: c_int = undefined;
    checkCuda(c.cuDeviceGetCount(&device_count), @src());

    checkCuda(c.cuDeviceGet(&device, 0), @src());
    std.log.info("Using device {d}", .{device});

    var name: [256]u8 = undefined;
    checkCuda(c.cuDeviceGetName(&name, name.len, device), @src());
    std.log.info("Device name: {s}", .{std.mem.sliceTo(&name, 0)});

    var major: c_int = undefined;
    var minor: c_int = undefined;
    checkCuda(c.cuDeviceComputeCapability(&major, &minor, device), @src());
    std.log.info("Compute capability: {d}.{d}", .{ major, minor });

    checkCuda(c.cuDeviceTotalMem(&totalMem, device), @src());
    std.log.info("Total memory: {d} GiB", .{GiB(totalMem)});

    checkCuda(c.cuCtxCreate(&context, 0, device), @src());
    std.log.info("Context created.", .{});

    checkCuda(c.cuModuleLoadData(&module, vecAdd), @src());
    std.log.info("Module loaded.", .{});

    checkCuda(c.cuModuleGetFunction(&function, module, "VecAdd"), @src());
    std.log.info("Function loaded.", .{});
}

fn finalize() void {
    checkCuda(c.cuCtxDestroy(context), @src());
}

fn setupDeviceMemory(
    d_a: *c.CUdeviceptr,
    d_b: *c.CUdeviceptr,
    d_c: *c.CUdeviceptr,
) void {
    checkCuda(c.cuMemAlloc_v2(d_a, N * @sizeOf(c_int)), @src());
    checkCuda(c.cuMemAlloc_v2(d_b, N * @sizeOf(c_int)), @src());
    checkCuda(c.cuMemAlloc_v2(d_c, N * @sizeOf(c_int)), @src());
}

fn releaseDeviceMemory(
    d_a: *c.CUdeviceptr,
    d_b: *c.CUdeviceptr,
    d_c: *c.CUdeviceptr,
) void {
    checkCuda(c.cuMemFree_v2(d_a.*), @src());
    checkCuda(c.cuMemFree_v2(d_b.*), @src());
    checkCuda(c.cuMemFree_v2(d_c.*), @src());
}

fn runKernel(
    d_a: *c.CUdeviceptr,
    d_b: *c.CUdeviceptr,
    d_c: *c.CUdeviceptr,
    n: c_int,
) void {
    var args: [4]?*anyopaque = .{
        @ptrCast(d_a),
        @ptrCast(d_b),
        @ptrCast(d_c),
        @constCast(@ptrCast(&n)),
    };

    checkCuda(
        c.cuLaunchKernel(function, N, 1, 1, 1, 1, 1, 0, null, &args, 0),
        @src(),
    );
}

inline fn checkCuda(err: c.CUresult, src: std.builtin.SourceLocation) void {
    __checkCudaError(err, src.file, src.line, src.column);
}

inline fn __checkCudaError(
    err: c.CUresult,
    file: []const u8,
    line: u32,
    column: u32,
) void {
    if (err != c.CUDA_SUCCESS) {
        var errName: [*c]const u8 = undefined;
        var errMsg: [*c]const u8 = undefined;

        _ = c.cuGetErrorName(err, &errName);
        _ = c.cuGetErrorString(err, &errMsg);

        std.log.err("{s}:{d}:{d}: {s} ({s})", .{
            file,
            line,
            column,
            errMsg,
            errName,
        });

        std.process.exit(1);
    }
}

pub fn main() !void {
    var d_a: c.CUdeviceptr = undefined;
    var d_b: c.CUdeviceptr = undefined;
    var d_c: c.CUdeviceptr = undefined;

    var a: [N]c_int = undefined;
    var b: [N]c_int = undefined;
    var c_buf: [N]c_int = undefined;

    init();

    for (0..N) |i| {
        a[i] = @intCast(i);
        b[i] = @intCast(i);
    }

    setupDeviceMemory(&d_a, &d_b, &d_c);

    std.log.info("Copying data to device.", .{});

    checkCuda(c.cuMemcpyHtoD_v2(d_a, &a, N * @sizeOf(c_int)), @src());
    checkCuda(c.cuMemcpyHtoD_v2(d_b, &b, N * @sizeOf(c_int)), @src());

    std.log.info("Running kernel.", .{});
    runKernel(&d_a, &d_b, &d_c, @intCast(N));

    checkCuda(c.cuCtxSynchronize(), @src());

    std.log.info("Copying data back from device.", .{});
    checkCuda(c.cuMemcpyDtoH_v2(&c_buf, d_c, N * @sizeOf(c_int)), @src());

    for (0..N) |i| {
        if (c_buf[i] != a[i] + b[i]) {
            std.log.info("Error at index {d}: {d} != {d}", .{ i, c_buf[i], a[i] + b[i] });
            std.process.exit(1);
        }
    }

    releaseDeviceMemory(&d_a, &d_b, &d_c);
    finalize();
}
