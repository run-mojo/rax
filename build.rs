extern crate gcc;

fn main() {
    // Build a pseudo-library so that we have symbols that we can link
    // against while building Rust code.
    gcc::Build::new()
        .file("c/rax.c")
        .file("c/rax_ext.c")
        .include("c/")
        .compile("librax.a");
}
