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
pub fn erfc<T>(x: T) -> T
where
    f64: std::convert::From<T>,
    T: std::convert::From<f64>,
{
    let x = f64::from(x);
    let y = unsafe { Faddeeva_erfc_re(x) };
    y.into()
}

/// the scaled complementary error function $\mathrm{erfcx}(z) = e^{z^2} \mathrm{erfc}(z) = w(iz)$
pub fn erfcx<T>(x: T) -> T
where
    f64: std::convert::From<T>,
    T: std::convert::From<f64>,
{
    let x = f64::from(x);
    let y = unsafe { Faddeeva_erfcx_re(x) };
    y.into()
}
