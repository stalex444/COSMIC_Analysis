"""
Integration tests for the optimized CMB analysis tests.

This module tests the optimized Scale Transition Test and Transfer Entropy Test
that were previously enhanced to prevent hanging and improve performance.
"""

import os
import pytest
import numpy as np
import tempfile
import time
try:
    from mock import patch, MagicMock  # Python 2.7
except ImportError:
    from unittest.mock import patch, MagicMock  # Python 3.x

# Try to import the modules to test
try:
    from wmap_cosmic_analysis.scale_transition_test import run_scale_transition_test
    from wmap_cosmic_analysis.transfer_entropy_test import run_transfer_entropy_test, calculate_transfer_entropy
except ImportError:
    # If the modules are not installed, mark all tests as skipped
    pytestmark = pytest.mark.skip(reason="wmap_cosmic_analysis modules not found")


@pytest.mark.integration
@pytest.mark.slow
class TestScaleTransitionTest:
    """Integration tests for the Scale Transition Test."""

    @pytest.fixture
    def mock_cmb_data(self):
        """Create mock CMB data for testing."""
        # Create a simple mock CMB dataset
        np.random.seed(12345)
        ells = np.arange(2, 1001)
        cls = 1000.0 * (ells / 10.0) ** (-2.0)
        # Add some noise
        cls += np.random.normal(0, cls * 0.1)
        return ells, cls

    def test_run_scale_transition_test_basic(self, mock_cmb_data, test_data_dir):
        """Test that the scale transition test runs without errors with basic options."""
        ells, cls = mock_cmb_data
        output_file = os.path.join(test_data_dir, "scale_transition_results.json")
        
        # Run with minimal simulations and timeout for faster testing
        try:
            result = run_scale_transition_test(
                ells=ells,
                cls=cls,
                output_file=output_file,
                num_simulations=3,  # Use very few simulations for testing
                timeout_seconds=10,
                visualize=False,
                max_clusters=3
            )
            
            # Check that the test ran and produced a result
            assert result is not None
            assert os.path.exists(output_file)
            
            # Check that the result contains expected keys
            assert 'p_value' in result
            assert 'clusters' in result
            assert 'significance' in result
            
        except Exception as e:
            pytest.fail("Scale transition test failed with error: {}".format(str(e)))

    def test_run_scale_transition_test_timeout(self, mock_cmb_data, test_data_dir):
        """Test that the scale transition test properly handles timeouts."""
        ells, cls = mock_cmb_data
        output_file = os.path.join(test_data_dir, "scale_transition_timeout_results.json")
        
        # Mock a function that would cause a timeout
        with patch('wmap_cosmic_analysis.scale_transition_test.perform_clustering', 
                  side_effect=lambda *args, **kwargs: time.sleep(2)):
            
            # Run with a very short timeout to trigger timeout handling
            result = run_scale_transition_test(
                ells=ells,
                cls=cls,
                output_file=output_file,
                num_simulations=2,
                timeout_seconds=1,  # Very short timeout
                visualize=False
            )
            
            # Check that the test handled the timeout gracefully
            assert result is not None
            assert 'timeout' in result
            assert result['timeout'] is True

    def test_run_scale_transition_test_early_stopping(self, mock_cmb_data, test_data_dir):
        """Test that the scale transition test implements early stopping."""
        ells, cls = mock_cmb_data
        output_file = os.path.join(test_data_dir, "scale_transition_early_stop_results.json")
        
        # Mock a highly significant result to trigger early stopping
        with patch('wmap_cosmic_analysis.scale_transition_test.calculate_p_value', 
                  return_value=0.001):  # Very significant p-value
            
            start_time = time.time()
            result = run_scale_transition_test(
                ells=ells,
                cls=cls,
                output_file=output_file,
                num_simulations=30,  # Would take longer without early stopping
                timeout_seconds=30,
                visualize=False,
                early_stopping=True
            )
            end_time = time.time()
            
            # Check that the test stopped early
            assert result is not None
            assert 'p_value' in result
            assert result['p_value'] <= 0.05  # Should be significant
            assert end_time - start_time < 10  # Should finish quickly due to early stopping

    def test_run_scale_transition_test_visualization(self, mock_cmb_data, test_data_dir):
        """Test that the scale transition test visualization option works."""
        ells, cls = mock_cmb_data
        output_file = os.path.join(test_data_dir, "scale_transition_viz_results.json")
        
        # Mock the plotting function to check if it's called
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            result = run_scale_transition_test(
                ells=ells,
                cls=cls,
                output_file=output_file,
                num_simulations=2,
                timeout_seconds=10,
                visualize=True,  # Enable visualization
                plot_dir=test_data_dir
            )
            
            # Check that the visualization was created
            assert mock_savefig.called


@pytest.mark.integration
@pytest.mark.slow
class TestTransferEntropyTest:
    """Integration tests for the Transfer Entropy Test."""

    @pytest.fixture
    def mock_time_series_data(self):
        """Create mock time series data for testing."""
        np.random.seed(67890)
        # Create two correlated time series
        t = np.linspace(0, 10, 500)
        x = np.sin(t) + np.random.normal(0, 0.1, size=len(t))
        y = np.sin(t + 0.5) + np.random.normal(0, 0.1, size=len(t))
        return x, y

    def test_calculate_transfer_entropy(self, mock_time_series_data):
        """Test the transfer entropy calculation function."""
        x, y = mock_time_series_data
        
        # Calculate transfer entropy
        te_xy = calculate_transfer_entropy(x, y)
        te_yx = calculate_transfer_entropy(y, x)
        
        # Check that the results are reasonable
        assert te_xy >= 0  # Transfer entropy should be non-negative
        assert te_yx >= 0
        assert isinstance(te_xy, float)
        assert isinstance(te_yx, float)

    def test_run_transfer_entropy_test_basic(self, mock_time_series_data, test_data_dir):
        """Test that the transfer entropy test runs without errors with basic options."""
        x, y = mock_time_series_data
        output_file = os.path.join(test_data_dir, "transfer_entropy_results.json")
        
        # Run with minimal simulations for faster testing
        try:
            result = run_transfer_entropy_test(
                x=x,
                y=y,
                output_file=output_file,
                num_simulations=3,  # Use very few simulations for testing
                timeout_seconds=10,
                visualize=False,
                max_data_points=100  # Limit data points for speed
            )
            
            # Check that the test ran and produced a result
            assert result is not None
            assert os.path.exists(output_file)
            
            # Check that the result contains expected keys
            assert 'transfer_entropy_xy' in result
            assert 'transfer_entropy_yx' in result
            assert 'p_value' in result
            assert 'significance' in result
            
        except Exception as e:
            pytest.fail("Transfer entropy test failed with error: {}".format(str(e)))

    def test_run_transfer_entropy_test_timeout(self, mock_time_series_data, test_data_dir):
        """Test that the transfer entropy test properly handles timeouts."""
        x, y = mock_time_series_data
        output_file = os.path.join(test_data_dir, "transfer_entropy_timeout_results.json")
        
        # Mock a function that would cause a timeout
        with patch('wmap_cosmic_analysis.transfer_entropy_test.calculate_transfer_entropy', 
                  side_effect=lambda *args, **kwargs: time.sleep(2)):
            
            # Run with a very short timeout to trigger timeout handling
            result = run_transfer_entropy_test(
                x=x,
                y=y,
                output_file=output_file,
                num_simulations=2,
                timeout_seconds=1,  # Very short timeout
                visualize=False
            )
            
            # Check that the test handled the timeout gracefully
            assert result is not None
            assert 'timeout' in result
            assert result['timeout'] is True

    def test_run_transfer_entropy_test_early_stopping(self, mock_time_series_data, test_data_dir):
        """Test that the transfer entropy test implements early stopping."""
        x, y = mock_time_series_data
        output_file = os.path.join(test_data_dir, "transfer_entropy_early_stop_results.json")
        
        # Mock a highly significant result to trigger early stopping
        with patch('wmap_cosmic_analysis.transfer_entropy_test.calculate_significance', 
                  return_value=(0.001, True)):  # Very significant result
            
            start_time = time.time()
            result = run_transfer_entropy_test(
                x=x,
                y=y,
                output_file=output_file,
                num_simulations=30,  # Would take longer without early stopping
                timeout_seconds=30,
                visualize=False,
                early_stopping=True
            )
            end_time = time.time()
            
            # Check that the test stopped early
            assert result is not None
            assert 'p_value' in result
            assert result['p_value'] <= 0.05  # Should be significant
            assert end_time - start_time < 10  # Should finish quickly due to early stopping

    def test_run_transfer_entropy_test_visualization(self, mock_time_series_data, test_data_dir):
        """Test that the transfer entropy test visualization option works."""
        x, y = mock_time_series_data
        output_file = os.path.join(test_data_dir, "transfer_entropy_viz_results.json")
        
        # Mock the plotting function to check if it's called
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            result = run_transfer_entropy_test(
                x=x,
                y=y,
                output_file=output_file,
                num_simulations=2,
                timeout_seconds=10,
                visualize=True,  # Enable visualization
                plot_dir=test_data_dir
            )
            
            # Check that the visualization was created
            assert mock_savefig.called

    def test_run_transfer_entropy_test_data_limit(self, test_data_dir):
        """Test that the transfer entropy test properly limits data points."""
        # Create a large dataset
        np.random.seed(12345)
        t = np.linspace(0, 100, 10000)  # Very large dataset
        x = np.sin(t) + np.random.normal(0, 0.1, size=len(t))
        y = np.sin(t + 0.5) + np.random.normal(0, 0.1, size=len(t))
        
        output_file = os.path.join(test_data_dir, "transfer_entropy_data_limit_results.json")
        
        # Run with data point limitation
        with patch('wmap_cosmic_analysis.transfer_entropy_test.calculate_transfer_entropy') as mock_calc_te:
            # Set up the mock to return a value and track calls
            mock_calc_te.return_value = 0.1
            
            run_transfer_entropy_test(
                x=x,
                y=y,
                output_file=output_file,
                num_simulations=2,
                timeout_seconds=10,
                visualize=False,
                max_data_points=500  # Limit to 500 data points
            )
            
            # Check that the function was called with limited data
            args, kwargs = mock_calc_te.call_args
            assert len(args[0]) <= 500  # First argument should be limited x data
            assert len(args[1]) <= 500  # Second argument should be limited y data
