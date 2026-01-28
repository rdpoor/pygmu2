#!/usr/bin/env python3
"""
Minimal test to explore DelayPE + CropPE interactions.

This test helps diagnose why pre-cropped items in SequencePE might still
be contributing values when they shouldn't.
"""

import numpy as np
from pygmu2 import ConstantPE, CropPE, DelayPE, Extent, NullRenderer

# Create a constant value (infinite extent)
const1 = ConstantPE(77.0)
print(f"const1 extent: {const1.extent()}")

# Crop it to [0, 1000)
cropped_const1 = CropPE(const1, Extent(0, 1000))
print(f"cropped_const1 extent: {cropped_const1.extent()}")

# Delay it by 0
delayed_const1 = DelayPE(cropped_const1, delay=0)
print(f"delayed_const1 extent: {delayed_const1.extent()}")

renderer = NullRenderer(sample_rate=44100)
renderer.set_source(delayed_const1)

# Test 1: Render within crop window (should get 77.0)
print("\n=== Test 1: Render at global time 500 (within crop window) ===")
snippet1 = delayed_const1.render(500, 10)
print(f"  First 5 values: {snippet1.data[:5, 0]}")
print(f"  All values are 77.0? {np.allclose(snippet1.data[:, 0], 77.0)}")

# Test 2: Render after crop window (should get 0.0)
print("\n=== Test 2: Render at global time 2500 (after crop window) ===")
snippet2 = delayed_const1.render(2500, 10)
print(f"  First 5 values: {snippet2.data[:5, 0]}")
print(f"  All values are 0.0? {np.allclose(snippet2.data[:, 0], 0.0)}")
print(f"  Max value: {np.max(snippet2.data)}")
print(f"  Min value: {np.min(snippet2.data)}")

# Test 3: Render exactly at crop boundary
print("\n=== Test 3: Render at global time 1000 (at crop boundary) ===")
snippet3 = delayed_const1.render(1000, 10)
print(f"  First 5 values: {snippet3.data[:5, 0]}")
print(f"  All values are 0.0? {np.allclose(snippet3.data[:, 0], 0.0)}")

# Test 4: Direct CropPE render (without DelayPE)
print("\n=== Test 4: Direct CropPE render at local time 2500 (after crop) ===")
snippet4 = cropped_const1.render(2500, 10)
print(f"  First 5 values: {snippet4.data[:5, 0]}")
print(f"  All values are 0.0? {np.allclose(snippet4.data[:, 0], 0.0)}")

# Test 5: What DelayPE requests from CropPE
print("\n=== Test 5: What DelayPE requests when rendering at global time 2500 ===")
print("  DelayPE.render(2500, 10) calls cropped_const1.render(2500 - 0, 10)")
print("  = cropped_const1.render(2500, 10)")
print("  CropPE should check: is 2500 >= 1000? Yes, so return zeros")

# Test 6: Multiple delayed cropped constants (simulating SequencePE)
print("\n=== Test 6: Multiple delayed cropped constants (SequencePE simulation) ===")
const2 = ConstantPE(70.0)
cropped_const2 = CropPE(const2, Extent(0, 1000))
delayed_const2 = DelayPE(cropped_const2, delay=2000)

from pygmu2 import MixPE
mixed = MixPE(delayed_const1, delayed_const2)

renderer.set_source(mixed)
snippet5 = mixed.render(2500, 10)
print(f"  At global time 2500:")
print(f"    delayed_const1 (delay=0, crop=[0,1000)): local time = 2500 (after crop) -> should be 0.0")
print(f"    delayed_const2 (delay=2000, crop=[0,1000)): local time = 500 (within crop) -> should be 70.0")
print(f"  Mixed result (first 5 values): {snippet5.data[:5, 0]}")
print(f"  Expected: [70.0, 70.0, 70.0, 70.0, 70.0]")
print(f"  All values are 70.0? {np.allclose(snippet5.data[:, 0], 70.0)}")
print(f"  Max value: {np.max(snippet5.data)}")
print(f"  Min value: {np.min(snippet5.data)}")
