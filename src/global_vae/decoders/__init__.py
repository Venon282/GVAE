"""decoders subpackage of global_vae.

Importing this package registers every built-in decoder implementation
(`1d_cnn_decoder_v1`, `1d_cnn_resnet_decoder_v1`) via each module's own
`@registerDecoder` decorator, mirroring `encoders/__init__.py` (see that
module's docstring for why this import is required).
"""

import global_vae.decoders.OneDCnnDecoder  # noqa: F401
import global_vae.decoders.OneDCnnResidualDecoder  # noqa: F401
