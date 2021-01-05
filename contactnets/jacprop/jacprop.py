import pdb  # noqa
import types
from typing import Callable, Dict

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module, ModuleList

from contactnets.utils import tensor_utils


class JacProp(Module):
    """Efficiently compute jacobian of a module.

    Works by maintaining a lookup dictionary mapping module types to their jacobian functions.
    A jacobian function takes as an input the self variable of the current module and the
    input to the module from the forwards propagation. It outputs the jacobian matrix of the
    module for that input.

    Attributes:
        jac_funcs: maps the module type to the described jacobian function. Can be extended to
        support additional types of modules before calling jacobian_enable.
    """
    jac_funcs: Dict[type, Callable[[Module, Tensor], Tensor]]

    def __init__(self) -> None:
        super().__init__()

        self.jac_funcs = dict()
        self.init_default_jac_funcs()

    def init_default_jac_funcs(self):
        self.jac_funcs[nn.Linear] = lambda module, forwards: \
            module.weight.unsqueeze(0).repeat(forwards.shape[0], 1, 1)

        self.jac_funcs[nn.ReLU] = lambda module, forwards: \
            tensor_utils.matrix_diag(torch.clamp(torch.sign(forwards.squeeze(1)), min=0))

        self.jac_funcs[nn.Tanh] = lambda module, forwards: \
            tensor_utils.matrix_diag(1 - torch.tanh(forwards.squeeze(1)) ** 2)

        def sequential_jac_func(module, forwards):
            x = forwards
            children = module.children()

            first_child = next(children)
            jac = first_child.compute_jacobian(x)
            x = first_child(x)

            for child in children:
                jac = child.compute_jacobian(x).bmm(jac)
                x = child(x)

            return jac

        self.jac_funcs[nn.modules.container.Sequential] = sequential_jac_func

    def jacobian_enable(self, module: Module) -> None:
        """Recursively enable jacobian computation on the module and its submodules.

        Registers a pre hook on the module to save the forwards pass value. If there is no
        compute_jacobian function, attempts to add it from the internal lookup dictionary.
        Then recurses on children() to do the same.

        Args:
            module: the module to enable jacobian computations on.
        """
        if not isinstance(module, ModuleList):
            # Save the forwards value in the module (may already be saved somewhere but idk)
            def save_forwards(module: Module, input: Tensor):
                module.forwards_value = input
            module.register_forward_pre_hook(save_forwards)

            jac_func = getattr(module, 'compute_jacobian', None)

            # Make sure there isn't some non-method attribute compute_jacobian
            assert not ((jac_func is not None) and (not callable(jac_func)))

            if jac_func is None:
                try:
                    new_jac_func = self.jac_funcs[type(module)]
                except Exception:
                    raise Exception(f'Cannot find jac_func handler for {type(module)}')

                module.compute_jacobian = types.MethodType(new_jac_func, module)  # type: ignore

        # Recursively enable jacobian computation for all module children
        for child in module.children():
            self.jacobian_enable(child)

    @staticmethod
    def jacobian(module: Module, x: Tensor) -> Tensor:
        """Compute the jacobian of the module for the input."""
        return module.compute_jacobian(x)  # type: ignore
