import math
import pytest
import torch
import torch.nn as nn
from models import SpectralConv2d, FNO2d


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def spec_conv(device):
    """Default SpectralConv2d with in=4, out=8, modes=(6, 6)."""
    return SpectralConv2d(in_channels=4, out_channels=8, modes1=6, modes2=6).to(device)

@pytest.fixture
def fno(device):
    """Default FNO2d matching the training defaults."""
    return FNO2d(modes1=16, modes2=16, width=32, num_layers=4).to(device)

class TestSpectralConv2d:
    def test_output_shape(self, spec_conv, device):
        """Output spatial dims must match input spatial dims."""
        x = torch.randn(2, 4, 32, 32, device=device)
        y = spec_conv(x)
        assert y.shape == (2, 8, 32, 32)

    def test_output_shape_non_square(self, device):
        """Works correctly with non-square spatial dimensions."""
        model = SpectralConv2d(3, 5, modes1=4, modes2=4).to(device)
        x = torch.randn(2, 3, 24, 40, device=device)
        y = model(x)
        assert y.shape == (2, 5, 24, 40)

    def test_output_is_real(self, spec_conv, device):
        """irfft2 output must be real-valued (not complex)."""
        x = torch.randn(1, 4, 32, 32, device=device)
        y = spec_conv(x)
        assert not y.is_complex()

    def test_weight_shapes(self):
        """Learnable weight tensors must have the correct shape."""
        model = SpectralConv2d(in_channels=3, out_channels=7, modes1=5, modes2=8)
        assert model.weights1.shape == (3, 7, 5, 8)
        assert model.weights2.shape == (3, 7, 5, 8)

    def test_weights_are_complex(self):
        """Weights are stored as complex floats for Fourier-space arithmetic."""
        model = SpectralConv2d(2, 4, 6, 6)
        assert model.weights1.dtype == torch.cfloat
        assert model.weights2.dtype == torch.cfloat

    def test_weights_are_parameters(self):
        """Both weight tensors must be registered as learnable parameters."""
        model = SpectralConv2d(2, 4, 6, 6)
        param_names = {name for name, _ in model.named_parameters()}
        assert "weights1" in param_names
        assert "weights2" in param_names

    def test_xavier_scale(self):
        """Glorot-style init: scale = sqrt(2 / (in + out))."""
        in_c, out_c = 4, 12
        model = SpectralConv2d(in_c, out_c, modes1=6, modes2=6)
        expected_scale = math.sqrt(2.0 / (in_c + out_c))
        assert abs(model.scale - expected_scale) < 1e-7

    def test_gradients_flow(self, spec_conv, device):
        """A backward pass must produce non-None gradients on both weights."""
        x = torch.randn(2, 4, 32, 32, device=device)
        loss = spec_conv(x).sum()
        loss.backward()
        assert spec_conv.weights1.grad is not None
        assert spec_conv.weights2.grad is not None

    def test_batch_size_one(self, spec_conv, device):
        """Batch size of 1 must not cause shape errors."""
        x = torch.randn(1, 4, 32, 32, device=device)
        y = spec_conv(x)
        assert y.shape == (1, 8, 32, 32)

    def test_modes_clamp_to_spatial_size(self, device):
        """Modes larger than half the spatial dim should still produce valid output."""
        # modes1=modes2=8, spatial=16 → rfft width = 9, so modes fits
        model = SpectralConv2d(2, 2, modes1=8, modes2=8).to(device)
        x = torch.randn(1, 2, 16, 16, device=device)
        y = model(x)
        assert y.shape == (1, 2, 16, 16)

    def test_different_batch_sizes_independent(self, spec_conv, device):
        """Output for a single sample must equal that sample's slice of a batched run."""
        torch.manual_seed(0)
        x = torch.randn(4, 4, 32, 32, device=device)
        y_batch = spec_conv(x)

        y_single = spec_conv(x[2:3])
        assert torch.allclose(y_batch[2:3], y_single, atol=1e-5)

class TestFNO2d:
    def test_output_shape(self, fno, device):
        """Output must be [B, H, W, 2] for the default 2-channel head."""
        x = torch.randn(2, 64, 64, 1, device=device)
        y = fno(x)
        assert y.shape == (2, 64, 64, 2)

    def test_amplitude_range(self, fno, device):
        """Amplitude channel (index 0) must lie strictly in [0, 1] (sigmoid output)."""
        x = torch.randn(4, 32, 32, 1, device=device)
        y = fno(x)
        amp = y[..., 0]
        assert amp.min() >= 0.0
        assert amp.max() <= 1.0

    def test_phase_range(self, fno, device):
        """Phase channel (index 1) must lie in [-π, π] (tanh * π output)."""
        x = torch.randn(4, 32, 32, 1, device=device)
        y = fno(x)
        phase = y[..., 1]
        assert phase.min() >= -math.pi - 1e-6
        assert phase.max() <= math.pi + 1e-6

    def test_output_is_real(self, fno, device):
        """Output tensor must be real-valued."""
        x = torch.randn(2, 32, 32, 1, device=device)
        y = fno(x)
        assert not y.is_complex()

    def test_spatial_dims_preserved(self, fno, device):
        """Spatial dimensions of input and output must match after padding removal."""
        for size in [32, 48, 64]:
            x = torch.randn(1, size, size, 1, device=device)
            y = fno(x)
            print(y.shape)
            assert y.shape[1] == size
            assert y.shape[2] == size

    def test_num_layers(self, device):
        """FNO2d must create exactly num_layers spectral/conv/norm modules."""
        for n in [1, 2, 4]:
            model = FNO2d(modes1=4, modes2=4, width=8, num_layers=n).to(device)
            assert len(model.spec_list) == n
            assert len(model.conv_list) == n
            assert len(model.norm_list) == n

    def test_gradients_flow(self, fno, device):
        """Backward pass must produce gradients throughout the model."""
        x = torch.randn(2, 32, 32, 1, device=device, requires_grad=False)
        y = fno(x)
        loss = y.sum()
        loss.backward()
        for name, param in fno.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_padding_frac_zero(self, device):
        """padding_frac=0 disables boundary padding and must still work."""
        model = FNO2d(modes1=4, modes2=4, width=8, num_layers=2, padding_frac=0.0).to(device)
        x = torch.randn(1, 16, 16, 1, device=device)
        y = model(x)
        assert y.shape == (1, 16, 16, 2)

    def test_no_nans_in_output(self, fno, device):
        """Forward pass on random input must not produce NaN or Inf values."""
        x = torch.randn(2, 32, 32, 1, device=device)
        y = fno(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_batch_size_one(self, fno, device):
        """Batch size of 1 is valid (InstanceNorm2d must not crash)."""
        x = torch.randn(1, 32, 32, 1, device=device)
        y = fno(x)
        assert y.shape == (1, 32, 32, 2)

    def test_spectral_layers_are_spectralconv2d(self, fno):
        """Every layer in spec_list must be an instance of SpectralConv2d."""
        for layer in fno.spec_list:
            assert isinstance(layer, SpectralConv2d)

    def test_conv_layers_are_conv2d(self, fno):
        """Every layer in conv_list must be a 1×1 Conv2d."""
        for layer in fno.conv_list:
            assert isinstance(layer, nn.Conv2d)
            assert layer.kernel_size == (1, 1)

    def test_norm_layers_are_instance_norm(self, fno):
        """Every layer in norm_list must be InstanceNorm2d."""
        for layer in fno.norm_list:
            assert isinstance(layer, nn.InstanceNorm2d)

    def test_activation_is_gelu(self, fno):
        """The model's activation must be GELU as specified in the architecture."""
        assert isinstance(fno.activation, nn.GELU)

    def test_train_eval_consistency(self, fno, device):
        """Switching between train/eval modes must not change output shape."""
        x = torch.randn(2, 32, 32, 1, device=device)
        fno.train()
        y_train = fno(x)
        fno.eval()
        with torch.no_grad():
            y_eval = fno(x)
        assert y_train.shape == y_eval.shape

    def test_parameter_count_scales_with_width(self, device):
        """A wider model must have strictly more parameters than a narrower one."""
        small = FNO2d(modes1=4, modes2=4, width=8, num_layers=2).to(device)
        large = FNO2d(modes1=4, modes2=4, width=32, num_layers=2).to(device)
        n_small = sum(p.numel() for p in small.parameters())
        n_large = sum(p.numel() for p in large.parameters())
        assert n_large > n_small


