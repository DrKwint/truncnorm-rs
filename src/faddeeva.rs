#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub fn erfcx(x: f64) -> f64 {
    unsafe { Faddeeva_erfcx_re(x) }
}

pub fn erfc(x: f64) -> f64 {
    unsafe { Faddeeva_erfc_re(x) }
}
