"""decoders subpackage of global_vae.

Importing this package registers every built-in decoder implementation
(`1d_cnn_decoder_v1`, `1d_cnn_resnet_decoder_v1`, `2d_cnn_decoder_v1`,
`2d_cnn_resnet_decoder_v1`, `1d_transformer_decoder_v1`, `2d_vit_decoder_v1`,
`shared_transformer_decoder_v1`) via each module's own `@registerDecoder`
decorator, mirroring `encoders/__init__.py` (see that module's docstring
for why this import is required).
"""

import global_vae.decoders.one_d_cnn_decoder  # noqa: F401
import global_vae.decoders.one_d_cnn_residual_decoder  # noqa: F401
import global_vae.decoders.one_d_transformer_decoder  # noqa: F401
import global_vae.decoders.shared_transformer_decoder  # noqa: F401
import global_vae.decoders.two_d_cnn_decoder  # noqa: F401
import global_vae.decoders.two_d_cnn_residual_decoder  # noqa: F401
import global_vae.decoders.two_d_vit_decoder  # noqa: F401
