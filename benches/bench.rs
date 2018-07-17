#![allow(dead_code)]
#![feature(lang_items)]
#![feature(test)]

extern crate test;
extern crate rax;

use rax::*;
use test::Bencher;

#[bench]
fn bench_replace(b: &mut Bencher) {
    let r = &mut RaxMap::<u64, u64>::new();
    for x in 0..4 {
        r.insert_null(x).expect("whoops!");
    }

    b.iter(move || {
        r.insert_null(3);
    });
}

#[bench]
fn bench_get(b: &mut Bencher) {
    let r = &mut RaxMap::<u64, u64>::new();
    for x in 0..2 {
        r.insert_null(x).expect("whoops!");
    }

    b.iter(move || {
        r.get(1);
    });
}