const std = @import("std");
const ArrayList = std.ArrayList;

const NVCC_Result = struct {
    lazy_path: std.Build.LazyPath,
    ptx_file: []const u8,
};

fn compileKernel(
    exe: *std.Build.Step.Compile,
    b: *std.Build,
    kernel: []const u8,
    arch: ?u32,
) !NVCC_Result {
    const nvcc_step = b.addSystemCommand(&.{"nvcc"});
    nvcc_step.addArg("-ptx");

    if (arch) |target| {
        nvcc_step.addArg(try std.fmt.allocPrint(b.allocator, "-arch=sm_{d}", .{target}));
    }

    nvcc_step.addArg(kernel);
    nvcc_step.addArg("-o");

    const base = std.mem.trimRight(u8, std.fs.path.basename(kernel), ".cu");
    const output_file = try std.fmt.allocPrint(b.allocator, "{s}.ptx", .{base});
    const output = nvcc_step.addOutputFileArg(output_file);

    exe.step.dependOn(&nvcc_step.step);

    return .{
        .lazy_path = output,
        .ptx_file = output_file,
    };
}

fn collectCudaFiles(
    b: *std.Build,
    exe: *std.Build.Step.Compile,
    dir_path: []const u8,
    arch: ?u32,
) !std.ArrayList(NVCC_Result) {
    var results = std.ArrayList(NVCC_Result).init(b.allocator);
    var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();

    var iter = dir.iterate();

    while (try iter.next()) |entry| {
        const full_path = try std.fs.path.join(b.allocator, &.{ dir_path, entry.name });
        defer b.allocator.free(full_path);

        switch (entry.kind) {
            .directory => {
                var sub_results = try collectCudaFiles(b, exe, full_path, arch);
                defer sub_results.deinit();
                for (sub_results.items) |item| {
                    try results.append(item);
                }
            },
            .file => {
                if (std.mem.endsWith(u8, entry.name, ".cu")) {
                    const result = try compileKernel(exe, b, full_path, arch);
                    try results.append(result);
                }
            },
            // TODO: Symlinks?
            else => {},
        }
    }

    return results;
}

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const cuda_include = b.option(
        []const u8,
        "cuda-include",
        "Path to cuda include",
    ) orelse "/opt/cuda/include";

    const kernels = b.option(
        []const u8,
        "kernels",
        "Path to kernels",
    ) orelse "src/kernels";

    const exe = b.addExecutable(.{
        .name = "cuda-zig-test",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .use_llvm = false,
        .use_lld = false,
    });

    const cuda_files = try collectCudaFiles(b, exe, kernels, 75);
    defer cuda_files.deinit();

    const translate_c = b.addTranslateC(.{
        .root_source_file = b.path("src/c.h"),
        .target = target,
        .optimize = optimize,
    });

    translate_c.addIncludePath(.{ .cwd_relative = cuda_include });

    exe.root_module.addImport("c", translate_c.createModule());

    exe.linkSystemLibrary("cuda");

    const no_bin = b.option(bool, "no-bin", "skip emitting binary") orelse false;
    if (no_bin) {
        b.getInstallStep().dependOn(&exe.step);
    } else {
        b.installArtifact(exe);
    }

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    for (cuda_files.items) |cuda_file| {
        exe.root_module.addAnonymousImport(
            cuda_file.ptx_file,
            .{ .root_source_file = cuda_file.lazy_path },
        );
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
