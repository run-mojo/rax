#![allow(dead_code)]
#![feature(lang_items)]
#![feature(test)]

extern crate libc;
extern crate test;

use std::error;
use std::fmt;
use std::mem::{size_of, transmute};
use std::ptr;

pub const GREATER: &'static str = ">";
pub const GREATER_EQUAL: &'static str = ">=";
pub const LESSER: &'static str = "<";
pub const LESSER_EQUAL: &'static str = "<=";
pub const EQUAL: &'static str = "=";
pub const BEGIN: &'static str = "^";
pub const END: &'static str = "$";

pub const RAX_NODE_MAX_SIZE: libc::c_int = ((1 << 29) - 1);
pub const RAX_STACK_STATIC_ITEMS: libc::c_int = 128;
pub const RAX_ITER_STATIC_LEN: libc::c_int = 128;
pub const RAX_ITER_JUST_SEEKED: libc::c_int = (1 << 0);
pub const RAX_ITER_EOF: libc::c_int = (1 << 1);
pub const RAX_ITER_SAFE: libc::c_int = (1 << 2);


#[derive(Debug)]
pub enum RaxError {
    Generic(GenericError),
    FromUtf8(std::string::FromUtf8Error),
    ParseInt(std::num::ParseIntError),
}

impl RaxError {
    pub fn generic(message: &str) -> RaxError {
        RaxError::Generic(GenericError::new(message))
    }
}

impl From<std::string::FromUtf8Error> for RaxError {
    fn from(err: std::string::FromUtf8Error) -> RaxError {
        RaxError::FromUtf8(err)
    }
}

impl From<std::num::ParseIntError> for RaxError {
    fn from(err: std::num::ParseIntError) -> RaxError {
        RaxError::ParseInt(err)
    }
}

impl fmt::Display for RaxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            // Both underlying errors already impl `Display`, so we defer to
            // their implementations.
            RaxError::Generic(ref err) => write!(f, "{}", err),
            RaxError::FromUtf8(ref err) => write!(f, "{}", err),
            RaxError::ParseInt(ref err) => write!(f, "{}", err),
        }
    }
}

impl error::Error for RaxError {
    fn description(&self) -> &str {
        // Both underlying errors already impl `Error`, so we defer to their
        // implementations.
        match *self {
            RaxError::Generic(ref err) => err.description(),
            RaxError::FromUtf8(ref err) => err.description(),
            RaxError::ParseInt(ref err) => err.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            // N.B. Both of these implicitly cast `err` from their concrete
            // types (either `&io::Error` or `&num::ParseIntError`)
            // to a trait object `&Error`. This works because both error types
            // implement `Error`.
            RaxError::Generic(ref err) => Some(err),
            RaxError::FromUtf8(ref err) => Some(err),
            RaxError::ParseInt(ref err) => Some(err),
        }
    }
}

#[derive(Debug)]
pub struct GenericError {
    message: String,
}

impl GenericError {
    pub fn new(message: &str) -> GenericError {
        GenericError {
            message: String::from(message),
        }
    }
}

impl<'a> fmt::Display for GenericError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Store error: {}", self.message)
    }
}

impl<'a> error::Error for GenericError {
    fn description(&self) -> &str {
        self.message.as_str()
    }

    fn cause(&self) -> Option<&error::Error> {
        None
    }
}



pub struct RaxSet<K: RaxKey> {
    rax: *mut rax,
    _marker: std::marker::PhantomData<K>,
}

/// Same as RaxMap except values are not pointers to heap allocations.
/// Instead the "data pointer" in the RAX is the value.
pub struct RaxIntMap<K: RaxKey>(RaxMap<K, usize>);

/// Redis has a beautiful Radix Tree implementation in ANSI C.
/// This brings it to Rust. Great effort went into this being zero overhead.
/// If you catch something that could be better go ahead and share it.
pub struct RaxMap<K: RaxKey, V> {
    pub rax: *mut rax,
    phantom: std::marker::PhantomData<(K, V)>,
}

impl<K: RaxKey, V> RaxMap<K, V> {
    pub fn new() -> RaxMap<K, V> {
        unsafe {
            RaxMap {
                rax: raxNew(),
                phantom: std::marker::PhantomData,
            }
        }
    }

    ///
    /// The number of entries in the RAX
    ///
    pub fn size(&self) -> u64 {
        unsafe { raxSize(self.rax) }
    }

    pub fn show(&self) {
        unsafe { raxShow(self.rax) }
    }

    ///
    /// Insert a new entry into the RAX
    ///
    pub fn insert_null(&mut self, key: K) -> Result<(i32, Option<Box<V>>), RaxError> {
        unsafe {
            // Allocate a pointer to catch the old value.
            let old: &mut *mut u8 = &mut ptr::null_mut();

            // Integer values require Big Endian to allow the Rax to fully optimize
            // storing them since it will be able to compress the prefixes especially
            // for 64/128bit numbers.
            let k = key.encode();
            let (ptr, len) = k.to_buf();

            let r = raxInsert(
                self.rax,
                // Grab a raw pointer to the key. Keys are most likely allocated
                // on the stack. The rax will keep it's own copy of the key so we
                // don't want to keep in in the heap twice and it exists in the
                // rax in it's compressed form.
                ptr,
                len,
                std::ptr::null_mut(),
                old,
            );

            // Was there an existing entry?
            if old.is_null() {
                Ok((r, None))
            } else {
                // Box the previous value since Rax is done with it and it's our
                // responsibility now to drop it. Once this Box goes out of scope
                // the value is dropped and memory reclaimed.
                Ok((r, Some(Box::from_raw(*old as *mut V))))
            }
        }
    }

    ///
    /// Insert a new entry into the RAX
    ///
    pub fn try_insert(&mut self, key: K, data: Box<V>) -> Result<(i32, Option<Box<V>>), RaxError> {
        unsafe {
            // Allocate a pointer to catch the old value.
            let old: &mut *mut u8 = &mut ptr::null_mut();

            // Leak the boxed value as we hand it over to Rax to keep track of.
            // These must be heap allocated unless we want to store sizeof(usize) or
            // less bytes, then the value can be the pointer.
            let value: &mut V = Box::leak(data);

            // Integer values require Big Endian to allow the Rax to fully optimize
            // storing them since it will be able to compress the prefixes especially
            // for 64/128bit numbers.
            let k = key.encode();
            let (ptr, len) = k.to_buf();

            let r = raxTryInsert(
                self.rax,
                // Grab a raw pointer to the key. Keys are most likely allocated
                // on the stack. The rax will keep it's own copy of the key so we
                // don't want to keep in in the heap twice and it exists in the
                // rax in it's compressed form.
                ptr,
                len,
                value as *mut V as *mut u8,
                old,
            );

            // Was there an existing entry?
            if old.is_null() {
                Ok((r, None))
            } else {
                // Box the previous value since Rax is done with it and it's our
                // responsibility now to drop it. Once this Box goes out of scope
                // the value is dropped and memory reclaimed.
                Ok((r, Some(Box::from_raw(*old as *mut V))))
            }
        }
    }

    ///
    /// Insert a new entry into the RAX
    ///
    pub fn insert(&mut self, key: K, data: Box<V>) -> Result<(i32, Option<Box<V>>), RaxError> {
        unsafe {
            // Allocate a pointer to catch the old value.
            let old: &mut *mut u8 = &mut ptr::null_mut();

            // Leak the boxed value as we hand it over to Rax to keep track of.
            // These must be heap allocated unless we want to store sizeof(usize) or
            // less bytes, then the value can be the pointer.
            let value: &mut V = Box::leak(data);

            // Integer values require Big Endian to allow the Rax to fully optimize
            // storing them since it will be able to compress the prefixes especially
            // for 64/128bit numbers.
            let k = key.encode();
            let (ptr, len) = k.to_buf();

            let r = raxInsert(
                self.rax,
                // Grab a raw pointer to the key. Keys are most likely allocated
                // on the stack. The rax will keep it's own copy of the key so we
                // don't want to keep in in the heap twice and it exists in the
                // rax in it's compressed form.
                ptr,
                len,
                value as *mut V as *mut u8,
                old,
            );

            // Was there an existing entry?
            if old.is_null() {
                Ok((r, None))
            } else {
                // Box the previous value since Rax is done with it and it's our
                // responsibility now to drop it. Once this Box goes out of scope
                // the value is dropped and memory reclaimed.
                Ok((r, Some(Box::from_raw(*old as *mut V))))
            }
        }
    }

    ///
    ///
    ///
    pub fn remove(&mut self, key: K) -> (bool, Option<Box<V>>) {
        unsafe {
            let old: &mut *mut u8 = &mut ptr::null_mut();
            let k = key.encode();
            let (ptr, len) = k.to_buf();

            let r = raxRemove(
                self.rax,
                ptr,
                len,
                old,
            );

            if old.is_null() {
                (r == 1, None)
            } else {
                (r == 1, Some(Box::from_raw(*old as *mut V)))
            }
        }
    }

    ///
    ///
    ///
    pub fn find_exists(&self, key: K) -> (bool, Option<&V>) {
        unsafe {
            let k = key.encode();
            let (ptr, len) = k.to_buf();

            let value = raxFind(
                self.rax,
                ptr,
                len,
            );

            if value.is_null() {
                (true, None)
            } else if value == raxNotFound {
                (false, None)
            } else {
                // transmute to the value so we don't drop the actual value accidentally.
                // While the key associated to the value is in the RAX then we cannot
                // drop it.
                (true, Some(transmute(value)))
            }
        }
    }

    ///
    ///
    ///
    pub fn find(&self, key: K) -> Option<&V> {
        unsafe {
            let k = key.encode();
            let (ptr, len) = k.to_buf();

            let value = raxFind(
                self.rax,
                ptr,
                len,
            );

            if value.is_null() || value == raxNotFound {
                None
            } else {
                // transmute to the value so we don't drop the actual value accidentally.
                // While the key associated to the value is in the RAX then we cannot
                // drop it.
                Some(std::mem::transmute(value))
            }
        }
    }

    ///
    ///
    ///
    pub fn exists(&self, key: K) -> bool {
        unsafe {
            let k = key.encode();
            let (ptr, len) = k.to_buf();

            let value = raxFind(
                self.rax,
                ptr,
                len,
            );

            if value.is_null() || value == raxNotFound {
                false
            } else {
                true
            }
        }
    }

    ///
    #[inline]
    pub fn seek_min<F>(
        &mut self,
        f: F,
    ) where
        F: Fn(
            &mut RaxMap<K, V>,
            &mut RaxCursor<K, V>,
        ) {
        unsafe {
            // Allocate stack memory.
            let mut cursor: RaxCursor<K, V> = std::mem::uninitialized();
            // Initialize a Rax iterator. This call should be performed a single time
            // to initialize the iterator, and must be followed by a raxSeek() call,
            // otherwise the raxPrev()/raxNext() functions will just return EOF.
            raxStart(&cursor as *const _ as *const raxIterator, self.rax);
            cursor.seek_min();
            // Borrow stack iterator and execute the closure.
            f(self, &mut cursor)
        }
    }

    ///
    #[inline]
    pub fn seek_min_result<R, F>(
        &mut self,
        op: &str,
        key: K,
        f: F,
    ) -> Result<R, RaxError>
        where
            F: Fn(
                &mut RaxMap<K, V>,
                &mut RaxCursor<K, V>,
            ) -> Result<R, RaxError> {
        unsafe {
            // Allocate stack memory.
            let mut cursor: RaxCursor<K, V> = std::mem::uninitialized();
            // Initialize a Rax iterator. This call should be performed a single time
            // to initialize the iterator, and must be followed by a raxSeek() call,
            // otherwise the raxPrev()/raxNext() functions will just return EOF.
            raxStart(&cursor as *const _ as *const raxIterator, self.rax);
            cursor.seek_min();
            // Borrow stack iterator and execute the closure.
            f(self, &mut cursor)
        }
    }

    ///
    #[inline]
    pub fn seek_max<F>(
        &mut self,
        f: F,
    ) where
        F: Fn(
            &mut RaxMap<K, V>,
            &mut RaxCursor<K, V>,
        ) {
        unsafe {
            // Allocate stack memory.
            let mut cursor: RaxCursor<K, V> = std::mem::uninitialized();
            // Initialize a Rax iterator. This call should be performed a single time
            // to initialize the iterator, and must be followed by a raxSeek() call,
            // otherwise the raxPrev()/raxNext() functions will just return EOF.
            raxStart(&cursor as *const _ as *const raxIterator, self.rax);
            cursor.seek_max();
            // Borrow stack iterator and execute the closure.
            f(self, &mut cursor)
        }
    }

    ///
    #[inline]
    pub fn seek_max_result<R, F>(
        &mut self,
        op: &str,
        key: K,
        f: F,
    ) -> Result<R, RaxError>
        where
            F: Fn(
                &mut RaxMap<K, V>,
                &mut RaxCursor<K, V>,
            ) -> Result<R, RaxError> {
        unsafe {
            // Allocate stack memory.
            let mut cursor: RaxCursor<K, V> = std::mem::uninitialized();
            // Initialize a Rax iterator. This call should be performed a single time
            // to initialize the iterator, and must be followed by a raxSeek() call,
            // otherwise the raxPrev()/raxNext() functions will just return EOF.
            raxStart(&cursor as *const _ as *const raxIterator, self.rax);
            cursor.seek_max();
            // Borrow stack iterator and execute the closure.
            f(self, &mut cursor)
        }
    }

    ///
    #[inline]
    pub fn seek<F>(
        &mut self,
        op: &str,
        key: K,
        f: F,
    ) where
        F: Fn(
            &mut RaxMap<K, V>,
            &mut RaxCursor<K, V>,
        ) {
        unsafe {
            // Allocate stack memory.
            let mut cursor: RaxCursor<K, V> = std::mem::uninitialized();
            // Initialize a Rax iterator. This call should be performed a single time
            // to initialize the iterator, and must be followed by a raxSeek() call,
            // otherwise the raxPrev()/raxNext() functions will just return EOF.
            raxStart(&cursor as *const _ as *const raxIterator, self.rax);
            cursor.seek(op, key);
            // Borrow stack iterator and execute the closure.
            f(self, &mut cursor)
        }
    }

    ///
    #[inline]
    pub fn seek_result<R, F>(
        &mut self,
        op: &str,
        key: K,
        f: F,
    ) -> Result<R, RaxError>
        where
            F: Fn(
                &mut RaxMap<K, V>,
                &mut RaxCursor<K, V>,
            ) -> Result<R, RaxError> {
        unsafe {
            // Allocate stack memory.
            let mut cursor: RaxCursor<K, V> = std::mem::uninitialized();
            // Initialize a Rax iterator. This call should be performed a single time
            // to initialize the iterator, and must be followed by a raxSeek() call,
            // otherwise the raxPrev()/raxNext() functions will just return EOF.
            raxStart(&cursor as *const _ as *const raxIterator, self.rax);
            cursor.seek(op, key);
            // Borrow stack iterator and execute the closure.
            f(self, &mut cursor)
        }
    }

    ///
    #[inline]
    pub fn iter<F>(&mut self, f: F) where F: Fn(&mut RaxMap<K, V>, &mut RaxCursor<K, V>) {
        unsafe {
            // Allocate stack memory.
            let mut i: RaxCursor<K, V> = std::mem::uninitialized();
            // Initialize a Rax iterator. This call should be performed a single time
            // to initialize the iterator, and must be followed by a raxSeek() call,
            // otherwise the raxPrev()/raxNext() functions will just return EOF.
            raxStart(&i as *const _ as *const raxIterator, self.rax);
            // Borrow stack iterator and execute the closure.
            f(self, &mut i)
        }
    }

    ///
    #[inline]
    pub fn iter_result<F, R>(
        &mut self, f: F,
    ) -> Result<R, RaxError>
        where
            F: Fn(&mut RaxMap<K, V>, &mut RaxCursor<K, V>) -> Result<R, RaxError> {
        unsafe {
            // Allocate stack memory.
            let mut i: RaxCursor<K, V> = std::mem::uninitialized();
            // Initialize a Rax iterator. This call should be performed a single time
            // to initialize the iterator, and must be followed by a raxSeek() call,
            // otherwise the raxPrev()/raxNext() functions will just return EOF.
            raxStart(&i as *const _ as *const raxIterator, self.rax);
            // Borrow stack iterator and execute the closure.
            f(self, &mut i)
        }
    }

    ///
    #[inline]
    pub fn iter_apply<F, R>(
        &mut self, f: F,
    ) -> Result<R, RaxError>
        where
            F: Fn(&mut RaxMap<K, V>, &mut RaxCursor<K, V>) -> Result<R, RaxError> {
        unsafe {
            // Allocate stack memory.
            let mut i: RaxCursor<K, V> = std::mem::uninitialized();
            // Initialize a Rax iterator. This call should be performed a single time
            // to initialize the iterator, and must be followed by a raxSeek() call,
            // otherwise the raxPrev()/raxNext() functions will just return EOF.
            raxStart(&i as *const _ as *const raxIterator, self.rax);
            // Borrow stack iterator and execute the closure.
            f(self, &mut i)
        }
    }
}

//
impl<K: RaxKey, V> Drop for RaxMap<K, V> {
    fn drop(&mut self) {
        unsafe {
            // Cleanup RAX
            raxFreeWithCallback(self.rax, RaxFreeWithCallbackWrapper::<V>);
        }
    }
}

pub trait RaxKey<RHS = Self>: Clone + Default + std::fmt::Debug {
    type Output: RaxKey;

    fn encode(self) -> Self::Output;

    fn to_buf(&self) -> (*const u8, usize);

    fn from_buf(ptr: *const u8, len: usize) -> RHS;
}

impl RaxKey for f32 {
    type Output = u32;

    #[inline]
    fn encode(self) -> Self::Output {
        // Encode as u32 Big Endian
        self.to_bits().to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        // This should never get called since we represent as a u32
        (self as *const _ as *const u8, std::mem::size_of::<f32>())
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> f32 {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe {
            // We used a BigEndian u32 to encode so let's reverse it
            f32::from_bits(
                u32::from_be(
                    *(ptr as *mut [u8; std::mem::size_of::<u32>()] as *mut u32)
                )
            )
        }
    }
}

impl RaxKey for f64 {
    type Output = u64;

    #[inline]
    fn encode(self) -> Self::Output {
        // Encode as u64 Big Endian
        self.to_bits().to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        // This should never get called since we represent as a u64
        (self as *const _ as *const u8, size_of::<f64>())
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> f64 {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe {
            // We used a BigEndian u64 to encode so let's reverse it
            f64::from_bits(
                u64::from_be(
                    *(ptr as *mut [u8; size_of::<u64>()] as *mut u64)
                )
            )
        }
    }
}

impl RaxKey for isize {
    type Output = isize;

    #[inline]
    fn encode(self) -> Self::Output {
        self.to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self as *const _ as *const u8, size_of::<isize>())
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> isize {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe { isize::from_be(*(ptr as *mut [u8; size_of::<isize>()] as *mut isize)) }
    }
}

impl RaxKey for usize {
    type Output = usize;

    #[inline]
    fn encode(self) -> Self::Output {
        self.to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self as *const _ as *const u8, std::mem::size_of::<usize>())
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> usize {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe { usize::from_be(*(ptr as *mut [u8; std::mem::size_of::<usize>()] as *mut usize)) }
    }
}

impl RaxKey for i16 {
    type Output = i16;

    #[inline]
    fn encode(self) -> Self::Output {
        self.to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self as *const _ as *const u8, 2)
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> i16 {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe { i16::from_be(*(ptr as *mut [u8; 2] as *mut i16)) }
    }
}

impl RaxKey for u16 {
    type Output = u16;

    #[inline]
    fn encode(self) -> Self::Output {
        self.to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self as *const _ as *const u8, 2)
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> u16 {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe { u16::from_be(*(ptr as *mut [u8; 2] as *mut u16)) }
    }
}

impl RaxKey for i32 {
    type Output = i32;

    #[inline]
    fn encode(self) -> Self::Output {
        self.to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self as *const _ as *const u8, 4)
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> i32 {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe { i32::from_be(*(ptr as *mut [u8; 4] as *mut i32)) }
    }
}

impl RaxKey for u32 {
    type Output = u32;

    #[inline]
    fn encode(self) -> Self::Output {
        self.to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self as *const _ as *const u8, 4)
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> u32 {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe { u32::from_be(*(ptr as *mut [u8; 4] as *mut u32)) }
    }
}

impl RaxKey for i64 {
    type Output = i64;

    #[inline]
    fn encode(self) -> Self::Output {
        self.to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self as *const _ as *const u8, 8)
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> i64 {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe { i64::from_be(*(ptr as *mut [u8; 8] as *mut i64)) }
    }
}

impl RaxKey for u64 {
    type Output = u64;

    #[inline]
    fn encode(self) -> Self::Output {
        self.to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self as *const _ as *const u8, 8)
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> u64 {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe { u64::from_be(*(ptr as *mut [u8; 8] as *mut u64)) }
    }
}

impl RaxKey for i128 {
    type Output = i128;

    #[inline]
    fn encode(self) -> Self::Output {
        self.to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self as *const _ as *const u8, 16)
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> i128 {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe { i128::from_be(*(ptr as *mut [u8; 16] as *mut i128)) }
    }
}

impl RaxKey for u128 {
    type Output = u128;

    #[inline]
    fn encode(self) -> Self::Output {
        self.to_be()
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self as *const _ as *const u8, 16)
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> u128 {
        if len != size_of::<Self>() {
            return Self::default();
        }
        unsafe { u128::from_be(*(ptr as *mut [u8; 16] as *mut u128)) }
    }
}

impl RaxKey for Vec<u8> {
    type Output = Vec<u8>;

    #[inline]
    fn encode(self) -> Vec<u8> {
        self
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self.as_ptr(), self.len())
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> Vec<u8> {
        Vec::from_buf(ptr, len)
    }
}

impl<'a> RaxKey for &'a [u8] {
    type Output = &'a [u8];

    #[inline]
    fn encode(self) -> &'a [u8] {
        self
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        (self.as_ptr(), self.len())
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

//impl RaxKey for SDS {
//    type Output = SDS;
//
//    #[inline]
//    fn encode(self) -> Self::Output {
//        self
//    }
//
//    #[inline]
//    fn to_buf(&self) -> (*const u8, usize) {
//        (self.as_ptr(), self.len())
//    }
//
//    #[inline]
//    fn from_buf(ptr: *const u8, len: usize) -> SDS {
//        SDS::from_ptr(ptr, len)
//    }
//}

impl<'a> RaxKey for &'a str {
    type Output = &'a str;

    #[inline]
    fn encode(self) -> Self::Output {
        self
    }

    #[inline]
    fn to_buf(&self) -> (*const u8, usize) {
        ((*self).as_ptr(), self.len())
    }

    #[inline]
    fn from_buf(ptr: *const u8, len: usize) -> &'a str {
        unsafe {
            std::str::from_utf8(
                std::slice::from_raw_parts(ptr, len)
            ).unwrap_or_default()
        }
    }
}

#[repr(C)]
pub struct RaxCursor<K: RaxKey, V> {
    pub flags: libc::c_int,
    pub rt: *mut rax,
    pub key: *mut u8,
    pub data: *mut libc::c_void,
    pub key_len: libc::size_t,
    pub key_max: libc::size_t,
    pub key_static_string: [u8; 128],
    pub node: *mut raxNode,
    pub stack: raxStack,
    pub node_cb: Option<raxNodeCallback>,
    _marker: std::marker::PhantomData<(K, V)>,
}

impl<K: RaxKey, V> Drop for RaxCursor<K, V> {
    fn drop(&mut self) {
        unsafe {
            raxStop(self as *const _ as *const raxIterator);
        }
    }
}

impl<K: RaxKey, V: 'static> Iterator for RaxCursor<K, V> {
    type Item = (K, Option<&'static V>);

    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        unsafe {
            if raxNext(self as *const _ as *const raxIterator) == 1 {
                let data: *mut libc::c_void = self.data;
                if data.is_null() {
                    None
                } else {
                    let val = data as *const V;
                    if val.is_null() {
                        Some((self.key(), None))
                    } else {
                        Some((self.key(), Some(std::mem::transmute(val as *mut u8))))
                    }
                }
            } else {
                None
            }
        }
    }
}

impl<K: RaxKey, V: 'static> DoubleEndedIterator for RaxCursor<K, V> {
    fn next_back(&mut self) -> Option<<Self as Iterator>::Item> {
        unsafe {
            if raxPrev(self as *const _ as *const raxIterator) == 1 {
                let data: *mut libc::c_void = self.data;
                if data.is_null() {
                    None
                } else {
                    let val = data as *const V;
                    if val.is_null() {
                        Some((self.key(), None))
                    } else {
                        Some((self.key(), Some(std::mem::transmute(val as *mut u8))))
                    }
                }
            } else {
                None
            }
        }
    }
}

impl<K: RaxKey, V> RaxCursor<K, V> {
    pub fn new(r: RaxMap<K, V>) -> RaxCursor<K, V> {
        unsafe {
            let mut iter: RaxCursor<K, V> = std::mem::uninitialized();
            raxStart(&mut iter as *mut _ as *mut raxIterator, r.rax);
            iter
        }
    }

    pub fn print_ptr(&self) {
        println!("ptr = {:p}", self);
        println!("ptr = {:p}", self as *const _ as *const raxIterator);
    }

    #[inline]
    pub fn begin(&self) -> bool {
        unsafe {
            raxSeek(
                self as *const _ as *const raxIterator,
                BEGIN.as_ptr(),
                std::ptr::null(),
                0,
            ) == 1
        }
    }

    #[inline]
    pub fn seek_min(&self) -> bool {
        unsafe {
            if raxSeek(
                self as *const _ as *const raxIterator,
                BEGIN.as_ptr(),
                std::ptr::null(),
                0,
            ) == 1 {
                self.forward()
            } else {
                false
            }
        }
    }

    #[inline]
    pub fn end(&self) -> bool {
        unsafe {
            raxSeek(
                self as *const _ as *const raxIterator,
                END.as_ptr(),
                std::ptr::null(),
                0,
            ) == 1
        }
    }

    #[inline]
    pub fn seek_max(&self) -> bool {
        unsafe {
            if raxSeek(
                self as *const _ as *const raxIterator,
                END.as_ptr(),
                std::ptr::null(),
                0,
            ) == 1 {
                self.back()
            } else {
                false
            }
        }
    }

    #[inline]
    pub fn back(&self) -> bool {
        unsafe {
            raxPrev(self as *const _ as *const raxIterator) == 1
        }
    }

    #[inline]
    pub fn go_prev(&self) -> bool {
        unsafe {
            raxPrev(self as *const _ as *const raxIterator) == 1
        }
    }

    #[inline]
    pub fn forward(&self) -> bool {
        unsafe {
            raxNext(self as *const _ as *const raxIterator) == 1
        }
    }

    #[inline]
    pub fn go_next(&self) -> bool {
        unsafe {
            raxNext(self as *const _ as *const raxIterator) == 1
        }
    }

    ///
    ///
    ///
    #[inline]
    pub fn key(&self) -> K {
        unsafe { K::from_buf(self.key, self.key_len as usize) }
    }

    #[inline]
    pub fn data(&self) -> Option<&V> {
        unsafe {
            let data: *mut libc::c_void = self.data;
            if data.is_null() {
                None
            } else {
                Some(std::mem::transmute(data as *mut u8))
            }
        }
    }

    #[inline]
    pub fn less_than(&self, key: K) -> bool {
        self.seek(LESSER, key)
    }

    #[inline]
    pub fn less_than_or_equal_to(&self, key: K) -> bool {
        self.seek(LESSER_EQUAL, key)
    }

    #[inline]
    pub fn greater_than(&self, key: K) -> bool {
        self.seek(GREATER, key)
    }

    #[inline]
    pub fn greater_than_or_equal_to(&self, key: K) -> bool {
        self.seek(GREATER_EQUAL, key)
    }

    #[inline]
    pub fn seek(&self, op: &str, key: K) -> bool {
        unsafe {
            let k = key.encode();
            let (p, len) = k.to_buf();
            raxSeek(
                self as *const _ as *const raxIterator,
                op.as_ptr(),
                p,
                len,
            ) == 1 && self.flags & RAX_ITER_EOF != 0
        }
    }

    #[inline]
    pub fn seek_raw(&self, op: &str, key: K) -> i32 {
        unsafe {
            let k = key.encode();
            let (p, len) = k.to_buf();
            raxSeek(self as *const _ as *const raxIterator, op.as_ptr(), p, len)
        }
    }

    #[inline]
    pub fn seek_bytes(&self, op: &str, ele: &[u8]) -> bool {
        unsafe {
            raxSeek(self as *const _ as *const raxIterator, op.as_ptr(), ele.as_ptr(), ele.len() as libc::size_t) == 1
        }
    }

    /// Return if the iterator is in an EOF state. This happens when raxSeek()
    /// failed to seek an appropriate element, so that raxNext() or raxPrev()
    /// will return zero, or when an EOF condition was reached while iterating
    /// with next() and prev().
    #[inline]
    pub fn eof(&self) -> bool {
        self.flags & RAX_ITER_EOF != 0
    }
}


#[derive(Clone, Copy)]
#[repr(C)]
pub struct rax;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct raxNode;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct raxStack {
    stack: *mut *mut libc::c_void,
    items: libc::size_t,
    maxitems: libc::size_t,
    static_items: [*mut libc::c_void; 32],
    oom: libc::c_int,
}

#[repr(C)]
pub struct raxIterator;

#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
extern "C" fn RaxFreeWithCallbackWrapper<V>(v: *mut libc::c_void) {
    unsafe {
        // Re-box it so it can drop it immediately after it leaves this scope.
        Box::from_raw(v as *mut V);
    }
}

#[allow(non_camel_case_types)]
type raxNodeCallback = extern "C" fn(v: *mut libc::c_void);


type RaxFreeCallback = extern "C" fn(v: *mut libc::c_void);

#[allow(improper_ctypes)]
#[allow(non_snake_case)]
#[allow(non_camel_case_types)]
#[link(name = "rax", kind = "static")]
extern "C" {
    #[no_mangle]
    pub static raxNotFound: *mut u8;

    // '>'
    #[no_mangle]
    pub static RAX_GREATER: *const u8;
    // '>='
    #[no_mangle]
    pub static RAX_GREATER_EQUAL: *const u8;
    // '<'
    #[no_mangle]
    pub static RAX_LESSER: *const u8;
    // '<='
    #[no_mangle]
    pub static RAX_LESSER_EQUAL: *const u8;
    // '='
    #[no_mangle]
    pub static RAX_EQUAL: *const u8;
    // '^'
    #[no_mangle]
    pub static RAX_MIN: *const u8;
    // '$'
    #[no_mangle]
    pub static RAX_MAX: *const u8;

    fn raxIteratorFree(
        it: *const raxIterator
    );

    fn raxIteratorSize() -> libc::c_int;

    fn raxNew() -> *mut rax;

    fn raxFree(
        rax: *mut rax
    );

    fn raxFreeWithCallback(
        rax: *mut rax,
        callback: RaxFreeCallback,
    );

    fn raxInsert(
        rax: *mut rax,
        s: *const u8,
        len: libc::size_t,
        data: *const u8,
        old: &mut *mut u8,
    ) -> libc::c_int;

    fn raxTryInsert(
        rax: *mut rax,
        s: *const u8,
        len: libc::size_t,
        data: *const u8,
        old: *mut *mut u8,
    ) -> libc::c_int;

    fn raxRemove(
        rax: *mut rax,
        s: *const u8,
        len: libc::size_t,
        old: &mut *mut u8,
    ) -> libc::c_int;

    fn raxFind(
        rax: *mut rax,
        s: *const u8,
        len: libc::size_t,
    ) -> *mut u8;

    fn raxIteratorNew(
        rt: *mut rax
    ) -> *mut raxIterator;

    fn raxStart(
        it: *const raxIterator,
        rt: *mut rax,
    );

    fn raxSeek(
        it: *const raxIterator,
        op: *const u8,
        ele: *const u8,
        len: libc::size_t,
    ) -> libc::c_int;

    fn raxNext(
        it: *const raxIterator
    ) -> libc::c_int;

    fn raxPrev(
        it: *const raxIterator
    ) -> libc::c_int;

    fn raxRandomWalk(
        it: *const raxIterator,
        steps: libc::size_t,
    ) -> libc::c_int;

    fn raxCompare(
        it: *const raxIterator,
        op: *const u8,
        key: *mut u8,
        key_len: libc::size_t,
    ) -> libc::c_int;

    fn raxStop(
        it: *const raxIterator
    );

    pub fn raxEOF(
        it: *const raxIterator
    ) -> libc::c_int;

    pub fn raxShow(
        rax: *mut rax
    );

    fn raxSize(
        rax: *mut rax
    ) -> libc::uint64_t;
}


#[cfg(test)]
mod tests {
    use *;
    use std;
    use std::default::Default;
    use std::fmt;
    use std::time::{Duration, Instant};
    use test::Bencher;

    pub struct MyMsg<'a>(&'a str);

    impl<'a> Drop for MyMsg<'a> {
        fn drop(&mut self) {
            println!("dropped -> {}", self.0);
        }
    }

    #[derive(Clone, Copy)]
    pub struct Stopwatch {
        start_time: Option<Instant>,
        elapsed: Duration,
    }

    impl Default for Stopwatch {
        fn default () -> Stopwatch {
            Stopwatch {
                start_time: None,
                elapsed: Duration::from_secs(0),
            }
        }
    }

    impl fmt::Display for Stopwatch {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            return write!(f, "{}ms", self.elapsed_ms());
        }
    }

    impl Stopwatch {
        pub fn new() -> Stopwatch {
            let sw: Stopwatch = Default::default();
            return sw;
        }
        pub fn start_new() -> Stopwatch {
            let mut sw = Stopwatch::new();
            sw.start();
            return sw;
        }

        pub fn start(&mut self) {
            self.start_time = Some(Instant::now());
        }
        pub fn stop(&mut self) {
            self.elapsed = self.elapsed();
            self.start_time = None;
        }
        pub fn reset(&mut self) {
            self.elapsed = Duration::from_secs(0);
            self.start_time = None;
        }
        pub fn restart(&mut self) {
            self.reset();
            self.start();
        }

        pub fn is_running(&self) -> bool {
            return self.start_time.is_some();
        }

        pub fn elapsed(&self) -> Duration {
            match self.start_time {
                Some(t1) => {
                    return t1.elapsed() + self.elapsed;
                },
                None => {
                    return self.elapsed;
                },
            }
        }
        pub fn elapsed_ms(&self) -> i64 {
            let dur = self.elapsed();
            return (dur.as_secs() * 1000 + (dur.subsec_nanos() / 1000000) as u64) as i64;
        }
    }

    #[bench]
    fn bench_fib(_b: &mut Bencher) {
        let r = &mut RaxMap::<u64, &str>::new();
        for x in 0..2000 {
            r.insert_null(x).expect("whoops!");
        }

        let sw = Stopwatch::start_new();

        for _po in 0..1000000 {
            r.find(300);
        }

        println!("Thing took {}ms", sw.elapsed_ms());
    }

    #[test]
    fn bench_tree() {
        for _ in 0..10 {
            let r = &mut std::collections::BTreeMap::<u64, &str>::new();
            for x in 0..2000 {
                r.insert(x, "");
            }

            let sw = Stopwatch::start_new();

            let xx = 300;
            for _po in 0..1000000 {
                r.get(&xx);
            }

            println!("Thing took {}ms", sw.elapsed_ms());
        }
    }

    #[test]
    fn bench_rax_find() {
        for _ in 0..10 {
            let r = &mut RaxMap::<u64, &str>::new();
            for x in 0..2000 {
                r.insert_null(x).expect("whoops!");
            }

            match r.find(1601) {
                Some(v) => println!("{}", v),
                None => {}
            }

            let sw = Stopwatch::start_new();

            for _po in 0..1000000 {
                r.find(1601);
            }

            println!("Thing took {}ms", sw.elapsed_ms());
        }
    }

    #[test]
    fn bench_rax_iter_find() {
        for _ in 0..10 {
            let r = &mut RaxMap::<u64, &str>::new();
            for x in 0..2000 {
                r.insert_null(x).expect("whoops!");
            }

            match r.find(1601) {
                Some(v) => println!("{}", v),
                None => {}
            }

            let sw = Stopwatch::start_new();

            for _po in 0..1000000 {
                r.iter(|_, iter| {
                    iter.seek(EQUAL, 1601);
                });
            }

            println!("Thing took {}ms", sw.elapsed_ms());
        }
    }

    #[test]
    fn bench_hash_find() {
        for _ in 0..10 {
            let r = &mut std::collections::HashMap::<u64, &str>::new();
//            r.insert_null(300);
            for x in 0..2000 {
                r.insert(x, "");
            }

            let sw = Stopwatch::start_new();

            let xx = 300;
            for _po in 0..1000000 {
                r.get(&xx);
            }

            println!("Thing took {}ms", sw.elapsed_ms());
        }
    }

    #[test]
    fn bench_rax_insert() {
        for _ in 0..10 {
            let mut r = &mut RaxMap::<u64, &str>::new();
//
            let sw = Stopwatch::start_new();

            for x in 0..1000000 {
                r.insert(x, Box::new("")).expect("whoops!");
            }

            println!("Thing took {}ms", sw.elapsed_ms());
            println!("Size {}", r.size());
        }
    }

    #[test]
    fn bench_rax_insert_show() {
        let r = &mut RaxMap::<u64, &str>::new();
//
        let sw = Stopwatch::start_new();

        for x in 0..1000 {
            r.insert(x, Box::new("")).expect("whoops!");
        }

        r.show();
        println!("Thing took {}ms", sw.elapsed_ms());
        println!("Size {}", r.size());
    }

    #[test]
    fn bench_rax_replace() {
        for _ in 0..10 {
            let mut r = &mut RaxMap::<u64, &str>::new();

            for x in 0..1000000 {
                r.insert(x, Box::new("")).expect("whoops!");
            }
//
            let sw = Stopwatch::start_new();

            for x in 0..1000000 {
                r.insert(x, Box::new("")).expect("whoops!");
            }

            println!("Thing took {}ms", sw.elapsed_ms());
            println!("Size {}", r.size());
        }
    }

    #[test]
    fn bench_tree_insert() {
        for _ in 0..10 {
            let mut r = &mut std::collections::BTreeMap::<u64, &str>::new();
//
            let sw = Stopwatch::start_new();

            for x in 0..1000000 {
                r.insert(x, "");
            }

            println!("Thing took {}ms", sw.elapsed_ms());
        }
    }

    #[test]
    fn bench_hashmap_insert() {
        for _ in 0..10 {
            let mut r = &mut std::collections::HashMap::<u64, &str>::new();
//
            let sw = Stopwatch::start_new();

            for x in 0..1000000 {
                r.insert(x, "");
            }

            println!("Thing took {}ms", sw.elapsed_ms());
            println!("Size {}", r.len());
        }
    }

    #[test]
    fn key_str() {
        let mut r = RaxMap::<&str, MyMsg>::new();

        let key = "hello-way";

        r.insert(
            key,
            Box::new(MyMsg("world 80")),
        ).expect("whoops!");
        r.insert(
            "hello-war",
            Box::new(MyMsg("world 80")),
        ).expect("whoops!");

        r.insert(
            "hello-wares",
            Box::new(MyMsg("world 80")),
        ).expect("whoops!");
        r.insert(
            "hello",
            Box::new(MyMsg("world 100")),
        ).expect("whoops!");

        {
            match r.find("hello") {
                Some(v) => println!("Found {}", v.0),
                None => println!("Not Found")
            }
        }

        r.show();

        r.iter(|_, iter| {
            iter.begin();
            while iter.forward() {
                println!("{}", iter.key());
            }
            iter.end();
            while iter.back() {
                println!("{}", iter.key());
            }
        });
    }

    #[test]
    fn key_f64() {
        println!("sizeof(Rax) {}", std::mem::size_of::<RaxMap<f64, MyMsg>>());

        let mut r = RaxMap::<f64, MyMsg>::new();

        r.insert(
            100.01,
            Box::new(MyMsg("world 100")),
        ).expect("whoops!");
        r.insert(
            80.20,
            Box::new(MyMsg("world 80")),
        ).expect("whoops!");
        r.insert(
            100.00,
            Box::new(MyMsg("world 200")),
        ).expect("whoops!");
        r.insert(
            99.10,
            Box::new(MyMsg("world 1")),
        ).expect("whoops!");

        r.show();

        r.iter(|_, iter| {
//            for (k, v) in iter {
//
//            }
            iter.begin();
            while iter.forward() {
                println!("{}", iter.key());
            }
            iter.end();
            while iter.back() {
                println!("{}", iter.key());
            }
        });
    }

    #[test]
    fn key_u64() {
        println!("sizeof(Rax) {}", std::mem::size_of::<RaxMap<u64, MyMsg>>());

        let mut r = RaxMap::<u64, MyMsg>::new();

        r.insert(
            100,
            Box::new(MyMsg("world 100")),
        ).expect("whoops!");
        r.insert(
            80,
            Box::new(MyMsg("world 80")),
        ).expect("whoops!");
        r.insert(
            200,
            Box::new(MyMsg("world 200")),
        ).expect("whoops!");
        r.insert(
            1,
            Box::new(MyMsg("world 1")),
        ).expect("whoops!");

        r.show();


//        let result = r.iter_result(move |it| {
//
//            if !it.seek(GREATER_EQUAL, 800) {
//                println!("Not Found");
//                return Ok("");
//            }
//
//            if it.eof() {
//                println!("Not Found");
//                return Ok("");
//            }
//
//            while it.forward() {
//                println!("Key Len = {}", it.key());
//                println!("Data = {}", it.data().unwrap().0);
//            }
//
//            Ok("")
//        });

//        r.seek(GREATER_EQUAL, 80, |_, iter| {
//            for (key, value) in iter {
//                println!("Key Len = {}", key);
//                println!("Data = {}", value.unwrap().0);
//            }
//        });

//        r.seek_result(GREATER_EQUAL, 80, |_, iter| {
//            for (key, value) in iter {
//                println!("Key Len = {}", key);
//                println!("Data = {}", value.unwrap().0);
//            }
//            Ok(())
//        });

        r.seek_min(|_, it| {
            for (key, value) in it.rev() {
                println!("Key Len = {}", key);
                unsafe { println!("Data = {}", value.unwrap().0); }
            }
        });

//        r.iter(move |it| {
//            if !it.seek(GREATER_EQUAL, 800) {
//                println!("Not Found");
//                return;
//            }
//
//
//
//            while it.forward() {
//                println!("Key Len = {}", it.key());
//                println!("Data = {}", it.data().unwrap().0);
//            }
//        });

//        let result = r.iter_apply(move |r, it| {
//            if !it.seek(GREATER_EQUAL, 800) {
//                println!("Out of Memory");
//                return Ok("");
//            }
//
//            r.insert(800, Box::new(MyMsg("moved")));
//            it.seek(GREATER_EQUAL, 800);
//
//            if it.eof() {
//                println!("Not Found");
//                return Ok("");
//            }
//
//            while it.back() {
//                println!("Key Len = {}", it.key());
//                println!("Data = {}", it.data().unwrap().0);
//            }
//
//            Ok("")
//        });
    }
}