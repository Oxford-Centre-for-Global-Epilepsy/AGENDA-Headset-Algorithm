import torch
import torch.nn.functional as F

class GradCAM:
    """
    Grad-CAM implementation for EEGNet.

    This class enables the generation of class activation heatmaps from a 
    specific convolutional layer in EEGNet by computing the gradients of 
    the class score with respect to the layer's feature maps.

    Typical usage:
        cam = GradCAM(model=eegnet, target_layer_name="separableConv")
        heatmap = cam.generate(input_tensor, class_idx=0)

    Attributes:
        model (nn.Module): The EEGNet model instance.
        target_layer (nn.Module): The convolutional layer to use for Grad-CAM.
        gradients (Tensor): Stores gradients after backward pass.
        activations (Tensor): Stores forward activations from the target layer.
    """

    def __init__(self, model, target_layer_name):
        """
        Initializes hooks on the target layer of the model to capture gradients
        and activations during the forward and backward passes.

        Args:
            model (nn.Module): A trained EEGNet model instance.
            target_layer_name (str): Name of the layer to visualize (e.g., "separableConv").
        """
        self.model = model.eval()
        self.target_layer = self._find_layer(target_layer_name)
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _find_layer(self, name):
        return dict([*self.model.eegnet.named_modules()])[name]

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        Generates a Grad-CAM heatmap for a given input and target class.

        Args:
            input_tensor (Tensor): EEG input tensor of shape [1, 1, C, T].
            class_idx (int, optional): Target class index to visualize.
                If None, the predicted class is used.

        Returns:
            heatmap (np.ndarray): 1D or 2D heatmap depending on the target layer's output shape.
                - Shape: [T'] or [F, T'], where T' is the downsampled time dimension.
        """
        output = self.model.eegnet(input_tensor)  # shape: [1, feature_dim]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.gradients  # [B, F, 1, T']
        activations = self.activations  # same shape

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # global avg pool → [B, F, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, 1, T']
        cam = F.relu(cam)
        cam = cam.squeeze()  # shape: [T'] or [F, T'] depending on layer

        cam -= cam.min()
        cam /= cam.max().clamp(min=1e-5)

        return cam.cpu().numpy()