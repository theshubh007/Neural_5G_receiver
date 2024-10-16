from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from MYSetup.helperLibrary import (
    StreamManagement,
    MyResourceGrid,
    Mapper,
    MyDemapper,
    MyResourceGridMapper,
    mygenerate_OFDMchannel,
    RemoveNulledSubcarriers,
    MyApplyOFDMChannel,
    MyApplyTimeChannel,
    OFDMModulator,
    OFDMDemodulator,
    BinarySource,
    ebnodb2no,
    complex_normal,
    time_lag_discrete_time_channel,
    cir_to_time_channel,
    hard_decisions,
    calculate_BER,
)

from sionna_tf import (
    MyLMMSEEqualizer,
)  # , OFDMDemodulator #ZFPrecoder, OFDMModulator, KroneckerPilotPattern, Demapper, RemoveNulledSubcarriers,
from channel import (
    LSChannelEstimator,
)  # , time_lag_discrete_time_channel #, ApplyTimeChannel #cir_to_time_channel

from Encode_Decode.ldpc import LDPC5GDecoder, LDPC5GEncoder


class Transmitter:
    def __init__(
        self,
        num_rx=1,
        num_tx=1,
        batch_size=1,
        fft_size=76,
        num_ofdm_symbols=14,
        num_bits_per_symbol=4,
        subcarrier_spacing=15e3,
        num_guard_carriers=[15, 16],
        pilot_ofdm_symbol_indices=[2],
        USE_LDPC=True,
        pilot_pattern="kronecker",
        guards=True,
        showfig=True,
    ) -> None:
        self.fft_size = fft_size
        self.batch_size = batch_size
        self.num_bits_per_symbol = num_bits_per_symbol
        self.showfig = showfig
        self.pilot_pattern = pilot_pattern

        # Generate random channel data
        self.channeldataset = RandomChannelDataset(num_rx=num_rx, num_tx=num_tx)

        # Load channel dataset using DataLoader with batch_size=1
        self.data_loader = DataLoader(
            dataset=self.channeldataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        h, tau = next(iter(self.data_loader))  # Get the first batch of random channels
        print("h shape:", h.shape)
        print("tau shape:", tau.shape)

        # torch dataloaders
        self.data_loader = DataLoader(
            dataset=self.channeldataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        if showfig:
            self.plotchimpulse()

        # The number of transmitted streams is equal to the number of UT antennas
        # in both uplink and downlink
        # NUM_STREAMS_PER_TX = NUM_UT_ANT
        # NUM_UT_ANT = num_rx
        num_streams_per_tx = num_rx  ##1
        # Create an RX-TX association matrix.
        # RX_TX_ASSOCIATION[i,j]=1 means that receiver i gets at least one stream
        # from transmitter j. Depending on the transmission direction (uplink or downlink),
        # the role of UT and BS can change.
        # For example, considering a system with 2 RX and 4 TX, the RX-TX
        # association matrix could be
        # [ [1 , 1, 0, 0],
        #   [0 , 0, 1, 1] ]
        # which indicates that the RX 0 receives from TX 0 and 1, and RX 1 receives from
        # TX 2 and 3.
        #
        # we have only a single transmitter and receiver,
        # the RX-TX association matrix is simply:
        # RX_TX_ASSOCIATION = np.array([[1]]) #np.ones([num_rx, 1], int)
        RX_TX_ASSOCIATION = np.ones([num_rx, num_tx], int)  # [[1]]
        self.STREAM_MANAGEMENT = StreamManagement(
            RX_TX_ASSOCIATION, num_streams_per_tx
        )  # RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX

        if guards:
            cyclic_prefix_length = 6  # 0 #6 Length of the cyclic prefix
            if num_guard_carriers is None and type(num_guard_carriers) is not list:
                num_guard_carriers = [
                    5,
                    6,
                ]  # [0, 0] #List of two integers defining the number of guardcarriers at the left and right side of the resource grid.
            dc_null = True  # False
            if (
                pilot_ofdm_symbol_indices is None
                and type(pilot_ofdm_symbol_indices) is not list
            ):
                pilot_ofdm_symbol_indices = [2, 11]
        else:
            cyclic_prefix_length = 0  # 0 #6 Length of the cyclic prefix
            num_guard_carriers = [
                0,
                0,
            ]  # List of two integers defining the number of guardcarriers at the left and right side of the resource grid.
            dc_null = False
            pilot_ofdm_symbol_indices = [0, 0]
        # pilot_pattern = "kronecker" #"kronecker", "empty"
        # fft_size = 76
        # num_ofdm_symbols=14
        RESOURCE_GRID = MyResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,  # 60e3, #30e3,
            num_tx=num_tx,  # 1
            num_streams_per_tx=num_streams_per_tx,  # 1
            cyclic_prefix_length=cyclic_prefix_length,
            num_guard_carriers=num_guard_carriers,
            dc_null=dc_null,
            pilot_pattern=pilot_pattern,
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )
        if showfig:
            RESOURCE_GRID.show()  # 14(OFDM symbol)*76(subcarrier) array=1064
            RESOURCE_GRID.pilot_pattern.show()

        if showfig and pilot_pattern == "kronecker":

            plt.figure()
            plt.title("Real Part of the Pilot Sequences")
            for i in range(num_streams_per_tx):
                plt.stem(
                    np.real(RESOURCE_GRID.pilot_pattern.pilots[0, i]),
                    markerfmt="C{}.".format(i),
                    linefmt="C{}-".format(i),
                    label="Stream {}".format(i),
                )
            plt.legend()
        print(
            "Average energy per pilot symbol: {:1.2f}".format(
                np.mean(np.abs(RESOURCE_GRID.pilot_pattern.pilots[0, 0]) ** 2)
            )
        )
        self.num_streams_per_tx = num_streams_per_tx
        self.RESOURCE_GRID = RESOURCE_GRID

        # num_bits_per_symbol = 4
        # Codeword length
        n = int(
            RESOURCE_GRID.num_data_symbols * num_bits_per_symbol
        )  # num_data_symbols:64*14=896 896*4=3584, if empty 1064*4=4256

        # USE_LDPC = True
        if USE_LDPC:
            coderate = 0.5
            # Number of information bits per codeword
            k = int(n * coderate)
            encoder = LDPC5GEncoder(k, n)  # 1824, 3648
            decoder = LDPC5GDecoder(encoder, hard_out=True)
            self.decoder = decoder
            self.encoder = encoder
        else:
            coderate = 1
            # Number of information bits per codeword
            k = int(n * coderate)
        self.k = k  # Number of information bits per codeword
        self.USE_LDPC = USE_LDPC
        self.coderate = coderate

        self.mapper = Mapper("qam", num_bits_per_symbol)
        self.rg_mapper = MyResourceGridMapper(
            RESOURCE_GRID
        )  # ResourceGridMapper(RESOURCE_GRID)

        # receiver part
        self.mydemapper = MyDemapper(
            "app", constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol
        )

    def plotchimpulse(self):
        h_b, tau_b = next(
            iter(self.data_loader)
        )  # h_b: [64, 1, 1, 1, 16, 10, 1], tau_b=[64, 1, 1, 10]
        # print(h_b.shape) #[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # print(tau_b.shape) #[batch, num_rx, num_tx, num_paths]
        tau_b = tau_b.numpy()  # torch tensor to numpy
        h_b = h_b.numpy()
        plt.figure()
        plt.title("Channel impulse response realization")
        plt.stem(
            tau_b[0, 0, 0, :] / 1e-9, np.abs(h_b)[0, 0, 0, 0, 0, :, 0]
        )  # 10 different pathes
        plt.xlabel(r"$\tau$ [ns]")
        plt.ylabel(r"$|a|$")

    def generateChannel(self, x_rg, no, channeltype="ofdm"):
        # x_rg:[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]

        h_b, tau_b = self.get_htau_batch()
        h_out = None
        # print(h_b.shape) #complex (64, 1, 1, 1, 16, 10, 1)[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # print(tau_b.shape) #float (64, 1, 1, 10)[batch, num_rx, num_tx, num_paths]
        if channeltype == "ofdm":
            # Generate the OFDM channel response
            # computes the Fourier transform of the continuous-time channel impulse response at a set of `frequencies`, corresponding to the different subcarriers.
            ##h: [64, 1, 1, 1, 16, 10, 1], tau: [64, 1, 1, 10] => (64, 1, 1, 1, 16, 1, 76)
            h_freq = mygenerate_OFDMchannel(
                h_b,
                tau_b,
                self.fft_size,
                subcarrier_spacing=60000.0,
                dtype=np.complex64,
                normalize_channel=True,
            )
            print("h_freq.shape", h_freq.shape)
            # h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            # (64, 1, 1, 1, 16, 1, 76)

            remove_nulled_scs = RemoveNulledSubcarriers(self.RESOURCE_GRID)
            h_out = remove_nulled_scs(h_freq)  # (64, 1, 1, 1, 16, 1, 64)
            if self.showfig:
                h_freq_plt = h_out[
                    0, 0, 0, 0, 0, 0
                ]  # get the last dimension: fft_size [76]
                # h_freq_plt = h_freq[0,0,0,0,0,0] #get the last dimension: fft_size [76]
                plt.figure()
                plt.plot(np.real(h_freq_plt))
                plt.plot(np.imag(h_freq_plt))
                plt.xlabel("Subcarrier index")
                plt.ylabel("Channel frequency response")
                plt.legend(["Ideal (real part)", "Ideal (imaginary part)"])
                plt.title("Comparison of channel frequency responses")

            # Generate the OFDM channel
            channel_freq = MyApplyOFDMChannel(add_awgn=True)
            # h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            # (64, 1, 1, 1, 16, 1, 76)
            y = channel_freq([x_rg, h_freq, no])  # h_freq is array
            # Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex
            # print(y.shape) #[64, 1, 1, 14, 76] dim (3,4 removed)

            # y = ApplyOFDMChannel(symbol_resourcegrid=x_rg, channel_frequency=h_freq, noiselevel=no, add_awgn=True)
            # y is the symbol received after the channel and noise
            # Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex
        elif channeltype == "perfect":
            y = x_rg
        elif channeltype == "awgn":
            y = x_rg  # (64, 1, 1, 14, 76)
            noise = complex_normal(y.shape, var=1.0)
            print(noise.dtype)
            noise = noise.astype(y.dtype)
            noise *= np.sqrt(no)
            y = y + noise
        elif channeltype == "time":
            bandwidth = self.RESOURCE_GRID.bandwidth  # 4560000
            l_min, l_max = time_lag_discrete_time_channel(bandwidth)  # -6, 20
            l_tot = l_max - l_min + 1  # 27
            # Compute the discrete-time channel impulse reponse
            h_time = cir_to_time_channel(
                bandwidth, h_b, tau_b, l_min=l_min, l_max=l_max, normalize=True
            )
            # h_time: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1] complex[64, 1, 1, 1, 16, 1, 27]
            h_out = h_time
            if self.showfig:
                plt.figure()
                plt.title("Discrete-time channel impulse response")
                plt.stem(np.abs(h_time[0, 0, 0, 0, 0, 0]))
                plt.xlabel(r"Time step $\ell$")
                plt.ylabel(r"$|\bar{h}|$")
            # channel_time = ApplyTimeChannel(self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=False)
            channel_time = MyApplyTimeChannel(
                self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=True
            )
            # OFDM modulator and demodulator
            modulator = OFDMModulator(self.RESOURCE_GRID.cyclic_prefix_length)
            demodulator = OFDMDemodulator(
                self.RESOURCE_GRID.fft_size,
                l_min,
                self.RESOURCE_GRID.cyclic_prefix_length,
            )

            # OFDM modulation with cyclic prefix insertion
            # x_rg:[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]
            x_time = modulator(x_rg)  # output: (64, 1, 1, 1064)
            # Compute the channel output
            # This computes the full convolution between the time-varying
            # discrete-time channel impulse reponse and the discrete-time
            # transmit signal. With this technique, the effects of an
            # insufficiently long cyclic prefix will become visible. This
            # is in contrast to frequency-domain modeling which imposes
            # no inter-symbol interfernce.
            y_time = channel_time([x_time, h_time, no])  # [64, 1, 1, 1174]
            # y_time = channel_time([x_time, h_time]) #(64, 1, 1, 1090) complex

            # Do modulator and demodulator test
            y_test = demodulator(x_time)
            differences = np.abs(x_rg - y_test)
            threshold = 1e-7
            num_differences = np.sum(differences > threshold)
            print("Number of differences:", num_differences)
            print(np.allclose(x_rg, y_test))
            print("Demodulation error (L2 norm):", np.linalg.norm(x_rg - y_test))

            # OFDM demodulation and cyclic prefix removal
            y = demodulator(y_time)
            # y = y_test
            # y: [64, 1, 1, 14, 76]
        return y, h_out

    def get_htau_batch(self, returnformat="numpy"):
        h_b, tau_b = next(
            iter(self.data_loader)
        )  # h_b: [64, 1, 1, 1, 16, 10, 1], tau_b=[64, 1, 1, 10]
        # print(h_b.shape) #[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # print(tau_b.shape) #[batch, num_rx, num_tx, num_paths]
        if returnformat == "numpy":
            tau_b = tau_b.numpy()  # torch tensor to numpy
            h_b = h_b.numpy()
        return h_b, tau_b

    def __call__(self, b=None, ebno_db=15.0, channeltype="ofdm", perfect_csi=False):
        # Transmitter
        if b is None:
            binary_source = BinarySource()
            # Start Transmitter self.k Number of information bits per codeword
            b = binary_source(
                [self.batch_size, 1, self.num_streams_per_tx, self.k]
            )  # [64,1,1,3584] if empty [64,1,1,1536] [batch_size, num_tx, num_streams_per_tx, num_databits]
        if self.USE_LDPC:
            c = self.encoder(
                b
            )  # tf.tensor[64,1,1,3072] [batch_size, num_tx, num_streams_per_tx, num_codewords]
        else:
            c = b
        x = self.mapper(
            c
        )  # np.array[64,1,1,896] if empty np.array[64,1,1,1064] 1064*4=4256 [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        x_rg = self.rg_mapper(x)  ##complex array[64,1,1,14,76] 14*76=1064
        # output: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]

        # set noise level
        ebnorange = np.linspace(-7, -5.25, 10)
        # ebno_db = 15.0
        # no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, RESOURCE_GRID)
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)
        # Convert it to a NumPy float
        no = np.float32(no)  # 0.0158

        y, h_out = self.generateChannel(x_rg, no, channeltype=channeltype)
        # y is the symbol received after the channel and noise
        # Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex
        print(
            y.shape
        )  # [64, 1, 1, 14, 76] dim (3,4 removed) h_out: (64, 1, 1, 1, 16, 1, 44)
        # print(y.real)
        # print(y.imag)
        print("Y:", type(y))
        print("Y shape:", y.shape)
        print("hout:", type(h_out))
        print("hout shape:", h_out.shape)
        print("b:", type(b))
        print("b shape:", b.shape)
        print(
            self.RESOURCE_GRID.pilot_pattern
        )  # <__main__.EmptyPilotPattern object at 0x7f2659dfd9c0>
        if (
            self.pilot_pattern == "empty" and perfect_csi == False
        ):  # "kronecker", "empty"
            # no channel estimation
            llr = self.mydemapper([y, no])  # [64, 1, 1, 14, 304]
            # Reshape the array by collapsing the last two dimensions
            llr_est = llr.reshape(llr.shape[:-2] + (-1,))  # (64, 1, 1, 4256)

            llr_perfect = self.mydemapper([x_rg, no])  # [64, 1, 1, 14, 304]
            llr_perfect = llr_perfect.reshape(
                llr_perfect.shape[:-2] + (-1,)
            )  # (64, 1, 1, 4256)
            b_perfect = hard_decisions(
                llr_perfect, np.int32
            )  ##(64, 1, 1, 4256) 0,1 [64, 1, 1, 14, 304] 2128
            # BER=calculate_BER(b, b_perfect)
            # print("Perfect BER:", BER)
        else:  # channel estimation or perfect_csi
            if perfect_csi == True:
                # For perfect CSI, the receiver gets the channel frequency response as input
                # However, the channel estimator only computes estimates on the non-nulled
                # subcarriers. Therefore, we need to remove them here from `h_freq` (done inside the self.generateChannel).
                # This step can be skipped if no subcarriers are nulled.
                h_hat, err_var = h_out, 0.0  # (64, 1, 1, 1, 16, 1, 64)
            else:  # perform channel estimation via pilots
                print(
                    "Num of Pilots:", len(self.RESOURCE_GRID.pilot_pattern.pilots)
                )  # 1
                # Receiver
                ls_est = LSChannelEstimator(
                    self.RESOURCE_GRID, interpolation_type="lin_time_avg"
                )
                # ls_est = MyLSChannelEstimator(self.RESOURCE_GRID, interpolation_type="lin_time_avg")

                # Observed resource grid y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], (64, 1, 1, 14, 76) complex
                # no : [batch_size, num_rx, num_rx_ant]
                print("no shape:", no.shape)
                print("y shape:", y.shape)
                print(no)
                h_hat, err_var = ls_est(
                    [y, no]
                )  # tf tensor (64, 1, 1, 1, 1, 14, 64), (1, 1, 1, 1, 1, 14, 64)
                # h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
                # Channel estimates accross the entire resource grid for all transmitters and streams

            if self.showfig:
                h_perfect = h_out[0, 0, 0, 0, 0, 0]  # (64, 1, 1, 1, 16, 1, 44)
                h_est = h_hat[0, 0, 0, 0, 0, 0]  # (64, 1, 1, 1, 1, 14, 44)
                plt.figure()
                plt.plot(np.real(h_perfect))
                plt.plot(np.imag(h_perfect))
                plt.plot(np.real(h_est), "--")
                plt.plot(np.imag(h_est), "--")
                plt.xlabel("Subcarrier index")
                plt.ylabel("Channel frequency response")
                plt.legend(
                    [
                        "Ideal (real part)",
                        "Ideal (imaginary part)",
                        "Estimated (real part)",
                        "Estimated (imaginary part)",
                    ]
                )
                plt.title("Comparison of channel frequency responses")

            # lmmse_equ = LMMSEEqualizer(self.RESOURCE_GRID, self.STREAM_MANAGEMENT)
            lmmse_equ = MyLMMSEEqualizer(self.RESOURCE_GRID, self.STREAM_MANAGEMENT)
            # input (y, h_hat, err_var, no)
            # Received OFDM resource grid after cyclic prefix removal and FFT y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
            # Channel estimates for all streams from all transmitters h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
            x_hat, no_eff = lmmse_equ(
                [y, h_hat, err_var, no]
            )  # (64, 1, 1, 912), (64, 1, 1, 912)
            # Estimated symbols x_hat : [batch_size, num_tx, num_streams, num_data_symbols], tf.complex
            # Effective noise variance for each estimated symbol no_eff : [batch_size, num_tx, num_streams, num_data_symbols], tf.float
            x_hat = x_hat.numpy()  # (64, 1, 1, 912)
            no_eff = no_eff.numpy()  # (64, 1, 1, 912)
            no_eff = np.mean(no_eff)

            llr_est = self.mydemapper([x_hat, no_eff])  # (64, 1, 1, 3072)
            # output: [batch size, num_rx, num_rx_ant, n * num_bits_per_symbol]

        # llr_est #(64, 1, 1, 4256)
        if self.USE_LDPC:
            b_hat_tf = self.decoder(llr_est)  # [64, 1, 1, 2128]
            b_hat = b_hat_tf.numpy()
        else:
            b_hat = hard_decisions(llr_est, np.int32)
        BER = calculate_BER(b, b_hat)
        print("BER Value:", BER)
        return b_hat, BER


# def get_deepMIMOdata(
#     scenario="O1_60",
#     # dataset_folder=r"D:\Dataset\CommunicationDataset\O1_60",
#     dataset_folder=r"D:\Research\neural_receiver\MYSetup\O1_60",
#     num_ue_antenna=1,
#     num_bs_antenna=16,
#     showfig=True,
# ):
#     # Load the default parameters
#     print("Loading DeepMIMO parameters...")
#     print(dataset_folder)
#     parameters = DeepMIMO.default_params()
#     # https://github.com/DeepMIMO/DeepMIMO-python/blob/master/src/DeepMIMOv3/params.py
#     # Set scenario name
#     parameters["scenario"] = scenario  # https://deepmimo.net/scenarios/o1-scenario/

#     # Set the main folder containing extracted scenarios
#     parameters["dataset_folder"] = (
#         dataset_folder  # r'D:\Dataset\CommunicationDataset\O1_60'
#     )

#     # To only include 10 strongest paths in the channel computation, num_paths integer in [1, 25]
#     parameters["num_paths"] = 10

#     # To activate only the first basestation, set
#     parameters["active_BS"] = np.array(
#         [1]
#     )  # Basestation indices to be included in the dataset
#     # parameters['active_BS'] = np.array([1, 5, 8]) #enable multiple basestations
#     # To activate the basestations 6, set
#     # parameters['active_BS'] = np.array([6])

#     parameters["OFDM"]["bandwidth"] = 0.05  # 50 MHz
#     print(parameters["OFDM"]["subcarriers"])  # 512
#     # parameters['OFDM']['subcarriers'] = 512 # OFDM with 512 subcarriers
#     # parameters['OFDM']['subcarriers_limit'] = 64 # Keep only first 64 subcarriers

#     # To activate the user rows 1-5, set
#     parameters["user_row_first"] = (
#         1  # 400 # First user row to be included in the dataset
#     )
#     parameters["user_row_last"] = (
#         100  # 450 # Last user row to be included in the dataset
#     )

#     # Configuration of the antenna arrays
#     parameters["bs_antenna"]["shape"] = np.array(
#         [num_bs_antenna, 1, 1]
#     )  # BS antenna shape through [x, y, z] axes
#     parameters["ue_antenna"]["shape"] = np.array(
#         [num_ue_antenna, 1, 1]
#     )  # UE antenna shape through [x, y, z] axes, single antenna

#     # The OFDM_channels parameter allows choosing between the generation of channel impulse
#     # responses (if set to 0) or frequency domain channels (if set to 1).
#     # It is set to 0 for this simulation, as the channel responses in frequency domain will be generated
#     parameters["OFDM_channels"] = 0

#     # Generate data
#     DeepMIMO_dataset = DeepMIMO.generate_data(parameters)

#     ## User locations
#     active_bs_idx = 0  # Select the first active basestation in the dataset
#     print(
#         DeepMIMO_dataset[active_bs_idx]["user"].keys()
#     )  # ['paths', 'LoS', 'location', 'distance', 'pathloss', 'channel']
#     print(
#         DeepMIMO_dataset[active_bs_idx]["user"]["location"].shape
#     )  # (9231, 3)  num_ue_locations: 9231
#     j = 0  # user j
#     print(
#         DeepMIMO_dataset[active_bs_idx]["user"]["location"][j]
#     )  # The Euclidian location of the user in the form of [x, y, z].

#     # Number of basestations
#     print(len(DeepMIMO_dataset))  # 1
#     # Keys of a basestation dictionary
#     print(DeepMIMO_dataset[0].keys())  # ['user', 'basestation', 'location']
#     # Keys of a channel
#     print(
#         DeepMIMO_dataset[0]["user"].keys()
#     )  # ['paths', 'LoS', 'location', 'distance', 'pathloss', 'channel']
#     # Number of UEs
#     print(len(DeepMIMO_dataset[0]["user"]["channel"]))  # 9231 18100
#     print(
#         DeepMIMO_dataset[active_bs_idx]["user"]["channel"].shape
#     )  # (num_ue_locations=18100, 1, bs_antenna=16, strongest_path=10)
#     # Shape of the channel matrix
#     print(DeepMIMO_dataset[0]["user"]["channel"].shape)  # (18100, 1, 16, 10)
#     # i=0
#     j = 0
#     print(DeepMIMO_dataset[active_bs_idx]["user"]["channel"][j])
#     # Float matrix of size (number of RX antennas) x (number of TX antennas) x (number of OFDM subcarriers)
#     # The channel matrix between basestation i and user j, Shape of BS 0 - UE 0 channel
#     print(DeepMIMO_dataset[active_bs_idx]["user"]["channel"][j].shape)  # (1, 16, 10)

#     # Path properties of BS 0 - UE 0
#     print(
#         DeepMIMO_dataset[active_bs_idx]["user"]["paths"][j]
#     )  # Ray-tracing Path Parameters in dictionary
#     #'num_paths': 9, Azimuth and zenith angle-of-arrivals – degrees (DoA_phi, DoA_theta), size of 9 array
#     # Azimuth and zenith angle-of-departure – degrees (DoD_phi, DoD_theta)
#     # Time of arrival – seconds (ToA)
#     # Phase – degrees (phase)
#     # Power – watts (power)
#     # Number of paths (num_paths)

#     print(
#         DeepMIMO_dataset[active_bs_idx]["user"]["LoS"][j]
#     )  # Integer of values {-1, 0, 1} indicates the existence of the LOS path in the channel.
#     # (1): The LoS path exists.
#     # (0): Only NLoS paths exist. The LoS path is blocked (LoS blockage).
#     # (-1): No paths exist between the transmitter and the receiver (Full blockage).

#     print(DeepMIMO_dataset[active_bs_idx]["user"]["distance"][j])
#     # The Euclidian distance between the RX and TX locations in meters.

#     print(DeepMIMO_dataset[active_bs_idx]["user"]["pathloss"][j])
#     # The combined path-loss of the channel between the RX and TX in dB.

#     print(DeepMIMO_dataset[active_bs_idx]["location"])
#     # Basestation Location [x, y, z].
#     print(DeepMIMO_dataset[active_bs_idx]["user"]["location"][j])
#     # The Euclidian location of the user in the form of [x, y, z].

#     # https://github.com/DeepMIMO/DeepMIMO-python/blob/master/src/DeepMIMOv3/sionna_adapter.py

#     if showfig:
#         plt.figure(figsize=(12, 8))
#         plt.scatter(
#             DeepMIMO_dataset[active_bs_idx]["user"]["location"][
#                 :, 1
#             ],  # y-axis location of the users
#             DeepMIMO_dataset[active_bs_idx]["user"]["location"][
#                 :, 0
#             ],  # x-axis location of the users
#             s=1,
#             marker="x",
#             c="C0",
#             label="The users located on the rows %i to %i (R%i to R%i)"
#             % (
#                 parameters["user_row_first"],
#                 parameters["user_row_last"],
#                 parameters["user_row_first"],
#                 parameters["user_row_last"],
#             ),
#         )  # 1-100
#         # First 181 users correspond to the first row
#         plt.scatter(
#             DeepMIMO_dataset[active_bs_idx]["user"]["location"][0:181, 1],
#             DeepMIMO_dataset[active_bs_idx]["user"]["location"][0:181, 0],
#             s=1,
#             marker="x",
#             c="C1",
#             label="First row of users (R%i)" % (parameters["user_row_first"]),
#         )

#         ## Basestation location
#         plt.scatter(
#             DeepMIMO_dataset[active_bs_idx]["location"][1],
#             DeepMIMO_dataset[active_bs_idx]["location"][0],
#             s=50.0,
#             marker="o",
#             c="C2",
#             label="Basestation",
#         )

#         plt.gca().invert_xaxis()  # Invert the x-axis to align the figure with the figure above
#         plt.ylabel("x-axis")
#         plt.xlabel("y-axis")
#         plt.grid()
#         plt.legend()

#         dataset = DeepMIMO_dataset
#         ## Visualization of a channel matrix
#         plt.figure()
#         # Visualize channel magnitude response
#         # First, select indices of a user and bs
#         ue_idx = 0
#         bs_idx = 0
#         # Import channel
#         channel = dataset[bs_idx]["user"]["channel"][ue_idx]
#         # Take only the first antenna pair
#         # plt.imshow(np.abs(np.squeeze(channel).T))
#         # plt.title('Channel Magnitude Response')
#         # plt.xlabel('TX Antennas')
#         # plt.ylabel('Subcarriers')

#         ## Visualization of the UE positions and path-losses
#         loc_x = dataset[bs_idx]["user"]["location"][:, 0]  # (9231,)
#         loc_y = dataset[bs_idx]["user"]["location"][:, 1]
#         loc_z = dataset[bs_idx]["user"]["location"][:, 2]
#         pathloss = dataset[bs_idx]["user"]["pathloss"]  # (9231,
#         fig = plt.figure()
#         ax = fig.add_subplot(projection="3d")
#         im = ax.scatter(loc_x, loc_y, loc_z, c=pathloss)
#         ax.set_xlabel("x (m)")
#         ax.set_ylabel("y (m)")
#         ax.set_zlabel("z (m)")

#         bs_loc_x = dataset[bs_idx]["basestation"]["location"][:, 0]
#         bs_loc_y = dataset[bs_idx]["basestation"]["location"][:, 1]
#         bs_loc_z = dataset[bs_idx]["basestation"]["location"][:, 2]
#         ax.scatter(bs_loc_x, bs_loc_y, bs_loc_z, c="r")
#         ttl = plt.title("UE and BS Positions")

#         fig = plt.figure()
#         ax = fig.add_subplot()
#         im = ax.scatter(loc_x, loc_y, c=pathloss)
#         ax.set_xlabel("x (m)")
#         ax.set_ylabel("y (m)")
#         fig.colorbar(im, ax=ax)
#         ttl = plt.title("UE Grid Path-loss (dB)")

#     return DeepMIMO_dataset


class RandomChannelDataset(Dataset):
    """Generates random channel data for the transmitter with specified dimensions."""

    def __init__(self, num_rx, num_tx):
        self.num_rx = num_rx
        self.num_tx = num_tx

    def __len__(self):
        return 1000  # Arbitrary length for the dataset

    def __getitem__(self, idx):
        # Generate random complex gains `h` with shape (1, 1, 1, 16, 10, 1)
        h = np.random.randn(1, 1, 1, 16, 10, 1) + 1j * np.random.randn(
            1, 1, 1, 16, 10, 1
        )

        # Generate random delays `tau` with shape (1, 1, 10)
        tau = np.random.rand(1, 1, 10)

        return h, tau


if __name__ == "__main__":

    showfigure = True

    transmit = Transmitter(
        num_rx=1,
        num_tx=1,
        batch_size=64,
        fft_size=76,
        num_ofdm_symbols=14,
        num_bits_per_symbol=4,
        subcarrier_spacing=60e3,
        USE_LDPC=False,
        pilot_pattern="kronecker",
        guards=True,
        showfig=showfigure,
    )  # "kronecker" "empty"
    # channeltype="perfect", "awgn", "ofdm", "time"
    b_hat, BER = transmit(ebno_db=5.0, channeltype="ofdm", perfect_csi=False)

    print("Finished")
