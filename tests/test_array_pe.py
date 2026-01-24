import unittest
import numpy as np

from pygmu2.array_pe import ArrayPE
from pygmu2.extent import Extent

class TestArrayPE(unittest.TestCase):

    def test_mono_array(self):
        """Test with 1D mono data."""
        data = [1.0, 0.5, 0.0]
        pe = ArrayPE(data)
        
        self.assertEqual(pe.channel_count(), 1)
        self.assertEqual(pe.extent(), Extent(0, 3))
        
        # Test full render
        snippet = pe.render(0, 3)
        np.testing.assert_array_equal(snippet.data, np.array([[1.0], [0.5], [0.0]], dtype=np.float32))
        
        # Test partial render
        snippet = pe.render(1, 1)
        np.testing.assert_array_equal(snippet.data, np.array([[0.5]], dtype=np.float32))

    def test_stereo_array(self):
        """Test with 2D stereo data."""
        data = [[1.0, -1.0], [0.5, -0.5], [0.0, 0.0]]
        pe = ArrayPE(data)
        
        self.assertEqual(pe.channel_count(), 2)
        
        snippet = pe.render(0, 3)
        np.testing.assert_array_equal(snippet.data, np.array(data, dtype=np.float32))

    def test_out_of_bounds(self):
        """Test output is zero outside array bounds."""
        data = [1.0, 2.0]
        pe = ArrayPE(data)
        
        # Before start
        snippet = pe.render(-1, 1)
        np.testing.assert_array_equal(snippet.data, np.zeros((1, 1), dtype=np.float32))
        
        # After end
        snippet = pe.render(2, 1)
        np.testing.assert_array_equal(snippet.data, np.zeros((1, 1), dtype=np.float32))
        
        # Overlapping end
        # Request 0 to 4 (array length 2) -> [1, 2, 0, 0]
        snippet = pe.render(0, 4)
        expected = np.array([[1.0], [2.0], [0.0], [0.0]], dtype=np.float32)
        np.testing.assert_array_equal(snippet.data, expected)

    def test_empty_array(self):
        """Test with empty data should raise ValueError."""
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            ArrayPE([])

    def test_numpy_input(self):
        """Test initialization with numpy array."""
        data = np.array([1, 2, 3], dtype=np.int16)
        pe = ArrayPE(data)
        
        # Should convert to float32
        snippet = pe.render(0, 3)
        self.assertEqual(snippet.data.dtype, np.float32)
        np.testing.assert_array_equal(snippet.data, np.array([[1], [2], [3]], dtype=np.float32))

if __name__ == "__main__":
    unittest.main()
