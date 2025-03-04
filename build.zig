const std = @import("std");

const NVCC_Result = struct {
    lazy: std.Build.LazyPath,
    file: []const u8,
};

fn compileKernel(
    exe: *std.Build.Step.Compile,
    b: *std.Build,
    kernel: []const u8,
) !NVCC_Result {
    const nvcc_step = b.addSystemCommand(&.{"nvcc"});
    nvcc_step.addArg("-ptx");
    nvcc_step.addArg("-arch=sm_75");
    nvcc_step.addArg(kernel);
    nvcc_step.addArg("-o");

    const base = std.mem.trimRight(u8, std.fs.path.basename(kernel), ".cu");
    const output_file = try std.fmt.allocPrint(b.allocator, "{s}.ptx", .{base});
    const output = nvcc_step.addOutputFileArg(output_file);

    exe.step.dependOn(&nvcc_step.step);

    return .{
        .lazy = output,
        .file = output_file,
    };
}

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "cuda-zig-test",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const path1 = try compileKernel(exe, b, "src/kernels/VecAdd.cu");
    const path2 = try compileKernel(exe, b, "src/kernels/matSum.cu");

    exe.addIncludePath(.{ .cwd_relative = "/opt/cuda/include/" });
    exe.addLibraryPath(.{ .cwd_relative = "/opt/cuda/lib/" });

    exe.linkSystemLibrary("cuda");
    exe.linkLibC();

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    exe.root_module.addAnonymousImport(path1.file, .{ .root_source_file = path1.lazy });
    exe.root_module.addAnonymousImport(path2.file, .{ .root_source_file = path2.lazy });

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
