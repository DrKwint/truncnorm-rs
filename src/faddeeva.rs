#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// the error function $\mathrm{erf}(z)$
pub fn erf(x: f64) -> f64 {
    unsafe { Faddeeva_erf_re(x) }
}

/// the complimentary error function $\mathrm{erfc}(z) = 1 - \mathrm{erf}(z)$
pub fn erfc(x: f64) -> f64 {
    unsafe { Faddeeva_erfc_re(x) }
}

/// the scaled complementary error function $\mathrm{erfcx}(z) = e^{z^2} \mathrm{erfc}(z) = w(iz)$
pub fn erfcx(x: f64) -> f64 {
    unsafe { Faddeeva_erfcx_re(x) }
}
