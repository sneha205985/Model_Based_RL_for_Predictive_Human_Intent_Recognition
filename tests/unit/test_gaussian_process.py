"""
Unit tests for Gaussian Process trajectory modeling.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock

from src.models.gaussian_process import (
    GaussianProcess,
    MultiOutputGP,
    GPParameters,
    RBFKernel,
    MaternKernel
)


class TestGPParameters:
    """Test cases for GPParameters data class."""
    
    def test_valid_gp_parameters(self):
        """Test creating valid GP parameters."""
        params = GPParameters(
            length_scale=1.0,
            output_scale=2.0,
            noise_variance=0.1,
            kernel_type='rbf'
        )
        
        assert params.length_scale == 1.0
        assert params.output_scale == 2.0
        assert params.noise_variance == 0.1
        assert params.kernel_type == 'rbf'
        assert params.ard is False
    
    def test_invalid_output_scale(self):
        """Test that invalid output scale raises error."""
        with pytest.raises(ValueError, match="Output scale must be positive"):
            GPParameters(
                length_scale=1.0,
                output_scale=-1.0,  # Invalid
                noise_variance=0.1
            )
    
    def test_invalid_noise_variance(self):
        """Test that invalid noise variance raises error."""
        with pytest.raises(ValueError, match="Noise variance must be positive"):
            GPParameters(
                length_scale=1.0,
                output_scale=1.0,
                noise_variance=-0.1  # Invalid
            )
    
    def test_invalid_length_scale_array(self):
        """Test that invalid length scale array raises error."""
        with pytest.raises(ValueError, match="All length scales must be positive"):
            GPParameters(
                length_scale=np.array([1.0, -0.5, 2.0]),  # Invalid
                output_scale=1.0,
                noise_variance=0.1
            )


class TestRBFKernel:
    """Test cases for RBF kernel."""
    
    @pytest.fixture
    def kernel(self):
        """Create RBF kernel instance."""
        return RBFKernel()
    
    @pytest.fixture
    def params(self):
        """Create test GP parameters."""
        return GPParameters(
            length_scale=1.0,
            output_scale=2.0,
            noise_variance=0.1,
            kernel_type='rbf'
        )
    
    def test_rbf_kernel_computation(self, kernel, params):
        """Test RBF kernel matrix computation."""
        X1 = np.array([[0.0], [1.0], [2.0]])
        X2 = np.array([[0.5], [1.5]])
        
        K = kernel(X1, X2, params)
        
        assert K.shape == (3, 2)
        assert np.all(K >= 0)  # RBF kernel is non-negative
        assert np.all(K <= params.output_scale)  # Bounded by output scale
        
        # Check symmetry for identical inputs
        K_sym = kernel(X1, X1, params)
        np.testing.assert_allclose(K_sym, K_sym.T, rtol=1e-10)
    
    def test_rbf_kernel_diagonal(self, kernel, params):
        """Test RBF kernel diagonal computation."""
        X = np.array([[0.0], [1.0], [2.0]])
        
        diag = kernel.diagonal(X, params)
        
        assert diag.shape == (3,)
        assert np.allclose(diag, params.output_scale)
    
    def test_rbf_kernel_properties(self, kernel, params):
        """Test RBF kernel mathematical properties."""
        X = np.array([[0.0], [1.0], [2.0]])
        
        K = kernel(X, X, params)
        
        # Should be positive definite
        eigenvals = np.linalg.eigvals(K)
        assert np.all(eigenvals > -1e-10)  # Allow small numerical errors
        
        # Diagonal should equal output scale
        assert np.allclose(np.diag(K), params.output_scale)
        
        # Should decrease with distance
        assert K[0, 0] >= K[0, 1] >= K[0, 2]
    
    def test_rbf_kernel_ard(self, kernel):
        """Test RBF kernel with ARD (multiple length scales)."""
        params = GPParameters(
            length_scale=np.array([1.0, 2.0]),
            output_scale=1.0,
            noise_variance=0.1,
            kernel_type='rbf',
            ard=True
        )
        
        X1 = np.array([[0.0, 0.0], [1.0, 2.0]])
        X2 = np.array([[0.5, 1.0], [1.5, 0.5]])
        
        K = kernel(X1, X2, params)
        
        assert K.shape == (2, 2)
        assert np.all(K >= 0)
        assert np.all(K <= params.output_scale)


class TestMaternKernel:
    """Test cases for Matern kernel."""
    
    @pytest.fixture
    def kernel_32(self):
        """Create Matern 3/2 kernel instance."""
        return MaternKernel(nu=1.5)
    
    @pytest.fixture
    def kernel_52(self):
        """Create Matern 5/2 kernel instance."""
        return MaternKernel(nu=2.5)
    
    @pytest.fixture
    def params(self):
        """Create test GP parameters."""
        return GPParameters(
            length_scale=1.0,
            output_scale=2.0,
            noise_variance=0.1,
            kernel_type='matern'
        )
    
    def test_matern32_kernel_computation(self, kernel_32, params):
        """Test Matern 3/2 kernel matrix computation."""
        X1 = np.array([[0.0], [1.0], [2.0]])
        X2 = np.array([[0.5], [1.5]])
        
        K = kernel_32(X1, X2, params)
        
        assert K.shape == (3, 2)
        assert np.all(K >= 0)
        assert np.all(K <= params.output_scale)
    
    def test_matern52_kernel_computation(self, kernel_52, params):
        """Test Matern 5/2 kernel matrix computation."""
        X1 = np.array([[0.0], [1.0], [2.0]])
        X2 = np.array([[0.5], [1.5]])
        
        K = kernel_52(X1, X2, params)
        
        assert K.shape == (3, 2)
        assert np.all(K >= 0)
        assert np.all(K <= params.output_scale)
    
    def test_matern_kernel_diagonal(self, kernel_32, params):
        """Test Matern kernel diagonal computation."""
        X = np.array([[0.0], [1.0], [2.0]])
        
        diag = kernel_32.diagonal(X, params)
        
        assert diag.shape == (3,)
        assert np.allclose(diag, params.output_scale)
    
    def test_matern_kernel_properties(self, kernel_32, params):
        """Test Matern kernel mathematical properties."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        
        K = kernel_32(X, X, params)
        
        # Should be positive definite
        eigenvals = np.linalg.eigvals(K)
        assert np.all(eigenvals > -1e-10)
        
        # Diagonal should equal output scale
        assert np.allclose(np.diag(K), params.output_scale)
        
        # Should be symmetric
        np.testing.assert_allclose(K, K.T, rtol=1e-10)
    
    def test_matern_different_nu_values(self, params):
        """Test Matern kernel with different nu values."""
        X = np.array([[0.0], [1.0]])
        
        # Test nu = 1.5
        kernel_15 = MaternKernel(nu=1.5)
        K_15 = kernel_15(X, X, params)
        
        # Test nu = 2.5
        kernel_25 = MaternKernel(nu=2.5)
        K_25 = kernel_25(X, X, params)
        
        # Both should be valid covariance matrices
        assert np.all(np.linalg.eigvals(K_15) > -1e-10)
        assert np.all(np.linalg.eigvals(K_25) > -1e-10)
        
        # They should be different (different smoothness)
        assert not np.allclose(K_15, K_25)


class TestGaussianProcess:
    """Test cases for GaussianProcess class."""
    
    @pytest.fixture
    def gp(self):
        """Create test GP instance."""
        return GaussianProcess(
            kernel_type='rbf',
            optimize_hyperparams=False,  # Disable for testing
            jitter=1e-6
        )
    
    @pytest.fixture
    def training_data(self):
        """Create test training data."""
        X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        y = np.sin(X).flatten()[:, None]  # Make 2D
        return X, y
    
    def test_gp_initialization(self, gp):
        """Test GP initialization."""
        assert gp.kernel_type == 'rbf'
        assert gp.optimize_hyperparams is False
        assert gp.is_fitted is False
        assert gp.X_train is None
        assert gp.y_train is None
    
    def test_gp_fit(self, gp, training_data):
        """Test GP fitting."""
        X, y = training_data
        
        fitted_gp = gp.fit(X, y)
        
        assert fitted_gp is gp  # Should return self
        assert gp.is_fitted is True
        assert gp.X_train.shape == X.shape
        assert gp.y_train.shape == y.shape
        assert gp.output_dim == 1
    
    def test_gp_fit_invalid_data(self, gp):
        """Test GP fitting with invalid data."""
        # Mismatched shapes
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0, 2.0])  # Wrong length
        
        with pytest.raises(ValueError, match="X and y must have same number of samples"):
            gp.fit(X, y)
    
    def test_gp_predict_before_fit(self, gp):
        """Test that prediction before fitting raises error."""
        X_test = np.array([[0.5]])
        
        with pytest.raises(RuntimeError, match="GP must be fitted before prediction"):
            gp.predict(X_test)
    
    def test_gp_predict(self, gp, training_data):
        """Test GP prediction."""
        X, y = training_data
        gp.fit(X, y)
        
        X_test = np.array([[0.25], [0.75], [1.25]])
        mean, std = gp.predict(X_test, return_std=True)
        
        assert mean.shape == (3, 1)
        assert std.shape == (3, 1)
        assert np.all(std >= 0)  # Standard deviation should be non-negative
    
    def test_gp_predict_no_std(self, gp, training_data):
        """Test GP prediction without standard deviation."""
        X, y = training_data
        gp.fit(X, y)
        
        X_test = np.array([[0.5]])
        mean = gp.predict(X_test, return_std=False)
        
        assert mean.shape == (1, 1)
        assert isinstance(mean, np.ndarray)
    
    def test_gp_predict_covariance(self, gp, training_data):
        """Test GP prediction with full covariance."""
        X, y = training_data
        gp.fit(X, y)
        
        X_test = np.array([[0.25], [0.75]])
        mean, cov = gp.predict(X_test, return_std=False, return_cov=True)
        
        assert mean.shape == (2, 1)
        assert cov.shape == (2, 2)
        
        # Covariance matrix should be positive semi-definite
        eigenvals = np.linalg.eigvals(cov)
        assert np.all(eigenvals >= -1e-10)
    
    def test_gp_log_marginal_likelihood(self, gp, training_data):
        """Test log marginal likelihood computation."""
        X, y = training_data
        gp.fit(X, y)
        
        log_ml = gp.log_marginal_likelihood()
        
        assert isinstance(log_ml, float)
        assert np.isfinite(log_ml)
    
    def test_gp_update(self, gp, training_data):
        """Test GP update with new data."""
        X, y = training_data
        gp.fit(X, y)
        
        # Add new data
        X_new = np.array([[2.5]])
        y_new = np.array([[np.sin(2.5)]])
        
        updated_gp = gp.update(X_new, y_new)
        
        assert updated_gp is gp
        assert gp.X_train.shape[0] == len(X) + len(X_new)
    
    def test_gp_hyperparameters(self, gp):
        """Test hyperparameter get/set methods."""
        original_params = gp.get_hyperparameters()
        
        assert 'length_scale' in original_params
        assert 'output_scale' in original_params
        assert 'noise_variance' in original_params
        
        # Modify parameters
        gp.set_hyperparameters(length_scale=2.0, output_scale=0.5)
        
        new_params = gp.get_hyperparameters()
        assert new_params['length_scale'] == 2.0
        assert new_params['output_scale'] == 0.5
    
    def test_gp_different_kernels(self, training_data):
        """Test GP with different kernel types."""
        X, y = training_data
        
        kernel_types = ['rbf', 'matern32', 'matern52']
        
        for kernel_type in kernel_types:
            gp = GaussianProcess(kernel_type=kernel_type, optimize_hyperparams=False)
            gp.fit(X, y)
            
            # Should be able to make predictions
            X_test = np.array([[0.5]])
            mean, std = gp.predict(X_test, return_std=True)
            
            assert mean.shape == (1, 1)
            assert std.shape == (1, 1)
            assert np.isfinite(mean[0, 0])
            assert std[0, 0] >= 0
    
    def test_gp_multidimensional_input(self, gp):
        """Test GP with multidimensional input."""
        # 2D input, 1D output
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        y = np.array([[0.0], [0.5], [1.0]])
        
        gp.fit(X, y)
        
        X_test = np.array([[0.25, 0.25], [0.75, 0.75]])
        mean, std = gp.predict(X_test, return_std=True)
        
        assert mean.shape == (2, 1)
        assert std.shape == (2, 1)
    
    @patch('src.models.gaussian_process.minimize')
    def test_gp_hyperparameter_optimization(self, mock_minimize, training_data):
        """Test GP hyperparameter optimization."""
        # Mock successful optimization
        mock_result = Mock()
        mock_result.success = True
        mock_result.x = [1.5, 2.5, 0.05]  # [length_scale, output_scale, noise_variance]
        mock_minimize.return_value = mock_result
        
        gp = GaussianProcess(kernel_type='rbf', optimize_hyperparams=True)
        X, y = training_data
        
        gp.fit(X, y)
        
        assert mock_minimize.called
        # Check that hyperparameters were updated
        params = gp.get_hyperparameters()
        assert params['length_scale'] == 1.5
        assert params['output_scale'] == 2.5
        assert params['noise_variance'] == 0.05


class TestMultiOutputGP:
    """Test cases for MultiOutputGP class."""
    
    @pytest.fixture
    def multi_gp(self):
        """Create multi-output GP instance."""
        return MultiOutputGP(
            kernel_type='rbf',
            optimize_hyperparams=False,
            n_outputs=3
        )
    
    @pytest.fixture
    def multi_training_data(self):
        """Create multi-output training data."""
        X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        y = np.column_stack([
            np.sin(X.flatten()),
            np.cos(X.flatten()),
            X.flatten() ** 2
        ])
        return X, y
    
    def test_multi_gp_initialization(self, multi_gp):
        """Test multi-output GP initialization."""
        assert multi_gp.n_outputs == 3
        assert len(multi_gp.gps) == 3
        assert multi_gp.is_fitted is False
    
    def test_multi_gp_fit(self, multi_gp, multi_training_data):
        """Test multi-output GP fitting."""
        X, y = multi_training_data
        
        fitted_gp = multi_gp.fit(X, y)
        
        assert fitted_gp is multi_gp
        assert multi_gp.is_fitted is True
        
        # All individual GPs should be fitted
        for gp in multi_gp.gps:
            assert gp.is_fitted is True
    
    def test_multi_gp_fit_wrong_outputs(self, multi_gp, multi_training_data):
        """Test multi-output GP with wrong number of outputs."""
        X, y = multi_training_data
        y_wrong = y[:, :2]  # Only 2 outputs instead of 3
        
        with pytest.raises(ValueError, match="Expected 3 outputs"):
            multi_gp.fit(X, y_wrong)
    
    def test_multi_gp_predict(self, multi_gp, multi_training_data):
        """Test multi-output GP prediction."""
        X, y = multi_training_data
        multi_gp.fit(X, y)
        
        X_test = np.array([[0.25], [0.75]])
        mean, std = multi_gp.predict(X_test, return_std=True)
        
        assert mean.shape == (2, 3)  # 2 test points, 3 outputs
        assert std.shape == (2, 3)
        assert np.all(std >= 0)
    
    def test_multi_gp_predict_no_std(self, multi_gp, multi_training_data):
        """Test multi-output GP prediction without std."""
        X, y = multi_training_data
        multi_gp.fit(X, y)
        
        X_test = np.array([[0.5]])
        mean = multi_gp.predict(X_test, return_std=False)
        
        assert mean.shape == (1, 3)
    
    def test_multi_gp_log_marginal_likelihood(self, multi_gp, multi_training_data):
        """Test multi-output GP log marginal likelihood."""
        X, y = multi_training_data
        multi_gp.fit(X, y)
        
        log_ml = multi_gp.log_marginal_likelihood()
        
        assert isinstance(log_ml, float)
        assert np.isfinite(log_ml)
    
    def test_multi_gp_update(self, multi_gp, multi_training_data):
        """Test multi-output GP update."""
        X, y = multi_training_data
        multi_gp.fit(X, y)
        
        # Add new data
        X_new = np.array([[2.5]])
        y_new = np.array([[np.sin(2.5), np.cos(2.5), 2.5**2]])
        
        updated_gp = multi_gp.update(X_new, y_new)
        
        assert updated_gp is multi_gp
        
        # All individual GPs should be updated
        for gp in multi_gp.gps:
            assert gp.X_train.shape[0] == len(X) + len(X_new)


@pytest.mark.unit  
class TestGPUtilities:
    """Test utility functions and edge cases."""
    
    def test_gp_with_single_point(self):
        """Test GP behavior with single training point."""
        gp = GaussianProcess(kernel_type='rbf', optimize_hyperparams=False)
        
        X = np.array([[1.0]])
        y = np.array([[2.0]])
        
        gp.fit(X, y)
        
        # Should be able to predict at training point
        mean, std = gp.predict(X, return_std=True)
        
        assert mean.shape == (1, 1)
        assert std.shape == (1, 1)
        # Should predict training value exactly (approximately)
        assert abs(mean[0, 0] - 2.0) < 0.1
        # Uncertainty at training point should be small
        assert std[0, 0] < 0.5
    
    def test_gp_with_duplicate_points(self):
        """Test GP with duplicate training points."""
        gp = GaussianProcess(kernel_type='rbf', optimize_hyperparams=False)
        
        X = np.array([[0.0], [0.0], [1.0], [1.0]])  # Duplicates
        y = np.array([[0.0], [0.1], [1.0], [0.9]])  # Slightly different y values
        
        # Should handle duplicates gracefully
        gp.fit(X, y)
        
        X_test = np.array([[0.5]])
        mean, std = gp.predict(X_test, return_std=True)
        
        assert mean.shape == (1, 1)
        assert std.shape == (1, 1)
        assert np.isfinite(mean[0, 0])
        assert std[0, 0] >= 0
    
    def test_gp_numerical_stability(self):
        """Test GP numerical stability with challenging data."""
        gp = GaussianProcess(kernel_type='rbf', optimize_hyperparams=False)
        
        # Very small length scale (potential numerical issues)
        gp.set_hyperparameters(length_scale=1e-10, noise_variance=1e-10)
        
        X = np.linspace(0, 1, 10)[:, None]
        y = np.sin(10 * X)
        
        # Should not crash
        gp.fit(X, y)
        
        X_test = np.array([[0.5]])
        mean, std = gp.predict(X_test, return_std=True)
        
        assert np.isfinite(mean[0, 0])
        assert np.isfinite(std[0, 0])
        assert std[0, 0] >= 0