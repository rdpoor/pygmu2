import unittest
import numpy as np

from pygmu2.trigger_pe import TriggerPE, TriggerMode, TriggerState
from pygmu2.snippet import Snippet
from pygmu2.processing_element import ProcessingElement
from pygmu2.array_pe import ArrayPE

class MockArrayPE(ProcessingElement):
    """Returns pre-defined data."""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)

    def inputs(self):
        return []

    def channel_count(self):
        return self.data.shape[1]

    def _render(self, start: int, duration: int) -> Snippet:
        # Return cyclic data or zeros if out of bounds? 
        # For tests, let's just return zeros if out of bounds to be safe,
        # or slice from the array.
        end = start + duration
        
        out = np.zeros((duration, self.data.shape[1]), dtype=np.float32)
        
        if start < len(self.data):
            avail = min(end, len(self.data)) - start
            out[:avail] = self.data[start : start + avail]
            
        return Snippet(start, out)

class MockRampPE(ProcessingElement):
    """Output = current time index (0, 1, 2...). Stateless, so pure."""
    def inputs(self):
        return []

    def channel_count(self):
        return 1

    def is_pure(self):
        return True  # Stateless: output depends only on (start, duration)

    def _render(self, start: int, duration: int) -> Snippet:
        data = np.arange(start, start + duration, dtype=np.float32).reshape(-1, 1)
        return Snippet(start, data)

class TestTriggerPE(unittest.TestCase):

    def test_idle(self):
        """Test silence when no trigger occurs."""
        source = MockRampPE()
        # Trigger is all zeros
        trigger = MockArrayPE([0.0] * 100)
        
        pe = TriggerPE(source, trigger, trigger_mode=TriggerMode.ONE_SHOT)
        
        result = pe.render(0, 10)
        np.testing.assert_array_equal(result.data, np.zeros((10, 1)))
        
        self.assertEqual(pe._state, TriggerState.ARMED)

    def test_one_shot_basic(self):
        """Test basic triggering in ONE_SHOT mode."""
        source = MockRampPE()
        # Trigger at index 2
        trigger_data = [0, 0, 1, 0, 0]
        trigger = MockArrayPE(trigger_data)
        
        pe = TriggerPE(source, trigger, trigger_mode=TriggerMode.ONE_SHOT)
        
        result = pe.render(0, 5)
        
        # Expected: 0, 0, source[0], source[1], source[2]
        # source[t] = t
        expected = np.array([[0], [0], [0], [1], [2]], dtype=np.float32)
        
        np.testing.assert_array_equal(result.data, expected)
        self.assertEqual(pe._state, TriggerState.ACTIVE)
        self.assertEqual(pe._start_time, 2)

    def test_one_shot_ignore_retrigger(self):
        """Test that subsequent triggers are ignored in ONE_SHOT mode."""
        source = MockRampPE()
        # Trigger at 2, and again at 4
        trigger_data = [0, 0, 1, 0, 1, 0]
        trigger = MockArrayPE(trigger_data)
        
        pe = TriggerPE(source, trigger, trigger_mode=TriggerMode.ONE_SHOT)
        
        result = pe.render(0, 6)
        
        # Expected: Start at 2. At 4, it should continue (source[2]), not restart (source[0]).
        # 0, 1: Silence
        # 2: source[0] = 0
        # 3: source[1] = 1
        # 4: source[2] = 2 (Ignored retrigger)
        # 5: source[3] = 3
        expected = np.array([[0], [0], [0], [1], [2], [3]], dtype=np.float32)
        
        np.testing.assert_array_equal(result.data, expected)

    def test_gated_cutoff(self):
        """Test that signal stops when gate goes low in GATED mode."""
        source = MockRampPE()
        # High at 2, Low at 4 (inclusive of low sample? "trigger <= 0")
        # 0: 0
        # 1: 0
        # 2: 1 (Start)
        # 3: 1 (Continue)
        # 4: 0 (Stop)
        trigger_data = [0, 0, 1, 1, 0, 0]
        trigger = MockArrayPE(trigger_data)
        
        pe = TriggerPE(source, trigger, trigger_mode=TriggerMode.GATED)
        
        result = pe.render(0, 6)
        
        # 0, 1: Silence
        # 2: source[0] = 0
        # 3: source[1] = 1
        # 4: Silence (Gate closed)
        # 5: Silence
        expected = np.array([[0], [0], [0], [1], [0], [0]], dtype=np.float32)
        
        np.testing.assert_array_equal(result.data, expected)
        self.assertEqual(pe._state, TriggerState.INACTIVE)

    def test_gated_no_retrigger(self):
        """Test that GATED does not retrigger after gate closes (one gate per session)."""
        source = MockRampPE()
        # 0: 0, 1: 1 (Start), 2: 0 (Stop), 3-4: no restart (stay silent)
        trigger_data = [0, 1, 0, 1, 1]
        trigger = MockArrayPE(trigger_data)
        
        pe = TriggerPE(source, trigger, trigger_mode=TriggerMode.GATED)
        
        result = pe.render(0, 5)
        
        # 0: Silence, 1: source[0]=0, 2: Silence (gate closed), 3-4: Silence (no retrigger)
        expected = np.array([[0], [0], [0], [0], [0]], dtype=np.float32)
        
        np.testing.assert_array_equal(result.data, expected)
        self.assertEqual(pe._state, TriggerState.INACTIVE)

    def test_block_boundary_continuation(self):
        """Test state preservation across render blocks."""
        source = MockRampPE()
        # Trigger at index 8 (inside first block of 10? No, let's render in chunks of 5)
        # We'll use a constant trigger of 0 then 1
        
        # Infinite trigger array effectively
        class StepTrigger(ProcessingElement):
            def inputs(self): return []
            def channel_count(self): return 1
            def _render(self, start, duration):
                # Trigger at t=3
                data = np.zeros(duration, dtype=np.float32)
                for i in range(duration):
                    if start + i >= 3:
                        data[i] = 1.0
                return Snippet(start, data.reshape(-1, 1))

        trigger = StepTrigger()
        pe = TriggerPE(source, trigger, trigger_mode=TriggerMode.ONE_SHOT)
        
        # Render first block (0-5)
        # Trigger at 3.
        # 0, 1, 2: Silence
        # 3: source[0] = 0
        # 4: source[1] = 1
        r1 = pe.render(0, 5)
        expected1 = np.array([[0], [0], [0], [0], [1]], dtype=np.float32)
        np.testing.assert_array_equal(r1.data, expected1)
        self.assertEqual(pe._state, TriggerState.ACTIVE)
        
        # Render second block (5-10)
        # Should continue: source[2]=2, source[3]=3...
        r2 = pe.render(5, 5)
        expected2 = np.array([[2], [3], [4], [5], [6]], dtype=np.float32)
        np.testing.assert_array_equal(r2.data, expected2)

    def test_one_shot_sample_accurate(self):
        """Test ONE_SHOT mode with ArrayPE for sample-accurate validation."""
        trigger = ArrayPE([0, 0, 1, 1, 0, 1, 1, 1, 0])
        signal = ArrayPE([10, 11, 12, 13, 14, 15, 16, 17, 18])
        
        triggered = TriggerPE(signal, trigger, trigger_mode=TriggerMode.ONE_SHOT)
        # rendering starts at sample index = 2
        snippet = triggered.render(0, 9)
        expected = np.array([[0.0], [0.0], [10], [11], [12], [13], [14], [15], [16]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)

    def test_gated_sample_accurate(self):
        """Test GATED mode with ArrayPE: gate opens at 2, closes at 4; no retrigger at 5-7."""
        trigger = ArrayPE([0, 0, 1, 1, 0, 1, 1, 1, 0])
        signal = ArrayPE([10, 11, 12, 13, 14, 15, 16, 17, 18])
        
        triggered = TriggerPE(signal, trigger, trigger_mode=TriggerMode.GATED)
        # Gate opens at 2, closes at 4; indices 5-7 stay silent (no retrigger)
        snippet = triggered.render(0, 9)
        expected = np.array([[0.0], [0.0], [10], [11], [0], [0], [0], [0], [0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)

    def test_retrigger_when_gate_high_again(self):
        """Test that RETRIGGER retriggers when gate goes high again after closing."""
        source = MockRampPE()
        trigger_data = [0, 1, 0, 1, 1]
        trigger = MockArrayPE(trigger_data)
        
        pe = TriggerPE(source, trigger, trigger_mode=TriggerMode.RETRIGGER)
        
        result = pe.render(0, 5)
        # 0: Silence, 1: source[0]=0, 2: Silence (gate closed), 3: source[0]=0 (retrigger), 4: source[1]=1
        expected = np.array([[0], [0], [0], [0], [1]], dtype=np.float32)
        
        np.testing.assert_array_equal(result.data, expected)
        self.assertEqual(pe._state, TriggerState.ACTIVE)

    def test_retrigger_sample_accurate(self):
        """Test RETRIGGER mode: gate opens at 2, closes at 4; retrigger at 5."""
        trigger = ArrayPE([0, 0, 1, 1, 0, 1, 1, 1, 0])
        signal = ArrayPE([10, 11, 12, 13, 14, 15, 16, 17, 18])
        
        triggered = TriggerPE(signal, trigger, trigger_mode=TriggerMode.RETRIGGER)
        snippet = triggered.render(0, 9)
        expected = np.array([[0.0], [0.0], [10], [11], [0], [10], [11], [12], [0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)

if __name__ == "__main__":
    unittest.main()
