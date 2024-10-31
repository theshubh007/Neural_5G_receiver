# from helperLibrary import (
#     hard_decisions,
# )
# import torch.nn as nn
# import numpy as np
# import torch
# from sionna.nr import TBDecoder, LayerDemapper, PUSCHLSChannelEstimator

# ls_est = PUSCHLSChannelEstimator(
#     resource_grid=rg,
#     dmrs_length=pc.dmrs.length,
#     dmrs_additional_position=pc.dmrs.additional_position,
#     num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
#     interpolation_type="nn",
# )
# class NeuralReceiver(nn.Module):

#     def __init__(
#         self,
#         resource_grid,
#         stream_management,
#         num_bits_per_symbol=4,
#         num_rx_ant=1,  # Number of antennas per receiver
#         num_tx=1,  # Number of receive antennas.
#     ):
#         super(NeuralReceiver, self).__init__()
#         self.resource_grid = resource_grid
#         self.stream_management = stream_management
#         self.num_bits_per_symbol = num_bits_per_symbol
#         self.num_rx_ant = num_rx_ant
#         self.num_tx = num_tx

#         # Define the channel estimator and equalizer
#         # Used convolutional neural network (CNN) for channel estimation and equalization
#         # spatial feature extraction
#         # The number of input channels is set to 2 * num_rx_ant, which means for each receive antenna (num_rx_ant), you have two input channels: one for the real part and one for the imaginary part of the received signal.

#         self.channel_estimator = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=2 * num_rx_ant, out_channels=128, kernel_size=3, padding=1
#             ),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#         )
#         self.equalizer = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, padding=1),
#             nn.Dropout(0.3),
#         )
#         # The demapper is a simple feedforward neural network
#         self.demapper = nn.Sequential(
#             nn.Linear(in_features=2, out_features=num_bits_per_symbol),
#             nn.ReLU(),
#             nn.Linear(
#                 in_features=num_bits_per_symbol, out_features=num_bits_per_symbol
#             ),
#         )

#         # Create the data mask to exclude pilots, guard bands, and DC null
#         num_ofdm_symbols = (
#             resource_grid.num_ofdm_symbols
#         )  # The number of OFDM symbols in the signal.
#         fft_size = (
#             resource_grid.fft_size
#         )  # The size of the FFT used in the OFDM modulation.
#         data_mask = np.ones((num_ofdm_symbols, fft_size), dtype=bool)

#         # Set guard subcarriers to False
#         left_guard = resource_grid.num_guard_carriers[0]
#         right_guard = resource_grid.num_guard_carriers[1]
#         data_mask[:, :left_guard] = False
#         data_mask[:, -right_guard:] = False

#         # Set DC null to False
#         if resource_grid.dc_null:
#             data_mask[:, fft_size // 2] = False

#         # Set pilot positions to False
#         pilot_pattern = resource_grid.pilot_pattern
#         pilot_ofdm_symbol_indices = pilot_pattern.pilot_ofdm_symbol_indices

#         # Assume pilot subcarriers span all frequencies for the specified OFDM symbols
#         for symbol_idx in pilot_ofdm_symbol_indices:
#             data_mask[symbol_idx, :] = (
#                 False  # Assume entire OFDM symbol is reserved for pilots
#             )

#         # Flatten data_mask to 1D for easy indexing later
#         self.data_mask_flat = data_mask.flatten()

#     def forward(self, y, no):
#         # y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex
#         if not torch.is_tensor(y):
#             y = torch.from_numpy(y)
#         y = y.type(torch.complex64)
#         print("Shape of y:", y.shape)

#         batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size = y.shape

#         # Combine real and imaginary parts into channels
#         y_real = y.real.type(torch.float32)
#         y_imag = y.imag.type(torch.float32)
#         y_input = torch.cat(
#             [y_real, y_imag], dim=2
#         )  # Shape: [batch_size, num_rx, num_rx_ant*2, num_ofdm_symbols, fft_size]
#         # print("Shape of y_input after concatenation:", y_input.shape)

#         y_input = torch.cat([y_real, y_imag], dim=2).view(
#             batch_size * num_rx, -1, num_ofdm_symbols, fft_size
#         )
#         # print("Shape of y_input after reshaping:", y_input.shape)

#         # Pass through the channel estimator
#         h_hat = self.channel_estimator(y_input)
#         # print("Shape of h_hat after channel estimator:", h_hat.shape)

#         # Pass through the equalizer
#         x_hat = self.equalizer(h_hat)
#         # print("Shape of x_hat after equalizer:", x_hat.shape)

#         # Reshape for demapping
#         x_hat = (
#             x_hat.permute(0, 2, 3, 1)
#             .contiguous()
#             .view(batch_size * num_rx, -1, 2)
#         )
#         # print("Shape of x_hat after permute:", x_hat.shape)
#         # x_hat = x_hat.view(batch_size, -1, 2)
#         # print("Shape of x_hat after reshaping for demapper:", x_hat.shape)

#         # Apply data mask to select only data subcarriers
#         data_mask = torch.from_numpy(self.data_mask_flat).to(x_hat.device)
#         x_hat_data = x_hat[:, data_mask]  # Only keep data subcarriers
#         # print("Shape of x_hat_data after masking:", x_hat_data.shape)

#         # Pass through the demapper
#         llr_est = self.demapper(x_hat_data)
#         # print("Shape of llr_est after demapper:", llr_est.shape)

#         # Reshape to match the expected output
#         # llr_est = llr_est.view(batch_size, 1, 1, -1)
#         # print("Shape of llr_est after final reshaping:", llr_est.shape)

#         # Convert llr_est to numpy for hard_decisions
#         llr_est_np = llr_est.detach().cpu().numpy()
#         # print("Shape of llr_est_np:", llr_est_np.shape)
#         # hard decision method is used to map the received LLR values into final bit values
#         b_hat = hard_decisions(llr_est_np, np.int32)
#         # print("Shape of b_hat:", b_hat.shape)

#         return b_hat
