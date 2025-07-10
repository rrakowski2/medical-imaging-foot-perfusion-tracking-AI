
# Script to help processing .csq files to image format - downloaded from: https://github.com/AlexanderProd/csq?tab=readme-ov-file

import re
import tempfile
import subprocess
import exiftool
from numpy import exp, sqrt, log
from libjpeg import decode
MAGIC_SEQ = re.compile(b"\x46\x46\x46\x00\x52\x54")


class CSQReader:
    def __init__(self, filename, blocksize=1000000):

        self.reader = open(filename, "rb")
        self.blocksize = blocksize
        self.leftover = b""
        self.imgs = []
        self.index = 0
        self.nframes = None
        self.et = exiftool.ExifTool()
        self.etHelper = exiftool.ExifToolHelper()
        self.et.run()

    def _populate_list(self):

        self.imgs = []
        self.index = 0

        x = self.reader.read(self.blocksize)
        if len(x) == 0:
            return

        matches = list(MAGIC_SEQ.finditer(x))
        if matches == []:
            return
        start = matches[0].start()

        if self.leftover != b"":
            self.imgs.append(self.leftover + x[:start])

        if matches[1:] == []:
            return

        for m1, m2 in zip(matches, matches[1:]):
            start = m1.start()
            end = m2.start()
            self.imgs.append(x[start:end])

        self.leftover = x[end:]

    def next_frame(self):

        if self.index >= len(self.imgs):
            self._populate_list()

            if len(self.imgs) == 0:
                return None

        im = self.imgs[self.index]

        raw, metadata = extract_data(im, self.etHelper)
        thermal_im = raw2temp(raw, metadata[0])
        self.index += 1

        return thermal_im

    def skip_frame(self):

        if self.index >= len(self.imgs):
            self._populate_list()

            if len(self.imgs) == 0:
                return False

        self.index += 1
        return True

    def count_frames(self):

        self.nframes = 0
        while self.skip_frame():
            self.nframes += 1
        self.reset()

        return self.nframes

    def frame_at(self, pos: int):

        if self.nframes == None:
            self.count_frames()

        if pos > self.nframes:
            print(f"File only has {self.nframes} frames.")
            return

        self.reset()
        fnum = 0
        while fnum < pos - 1:
            self.skip_frame()
            fnum += 1

        return self.next_frame()

    def frames(self):

        for im in self.imgs:
            self.index += 1
            if self.index >= len(self.imgs):
                self._populate_list()
                yield from self.frames()

            raw, metadata = extract_data(im, self.etHelper)
            thermal_im = raw2temp(raw, metadata[0])

            yield thermal_im

    def get_metadata(self):

        if self.index >= len(self.imgs):
            self._populate_list()

            if len(self.imgs) == 0:
                return None

        im = self.imgs[self.index]

        _, metadata = extract_data(im, self.etHelper)

        return metadata

    def reset(self):
        self.reader.seek(0)

    def close(self):
        self.reader.close()

def extract_data(bin, etHelper):
    # Create a temporary file for ExifTool to process
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        fp.write(bin)
        fp.flush()
        fname = fp.name

    # Attempt to retrieve metadata
    try:
        metadata = etHelper.get_metadata(fname)
        # Debug: Metadata retrieval (can be commented out for production)
        # print(f"Debug: Metadata retrieved: {metadata}")

    except Exception as e:
        print(f"Error retrieving metadata: {e}")
        raise

    # Extract raw thermal image
    try:
        binary = subprocess.check_output(["exiftool", "-b", "-RawThermalImage", fname])
        # Debug: Binary data length (can be commented out for production)
        # print(f"Debug: Binary data length: {len(binary)}")
        # Write binary data for troubleshooting (optional, remove if not needed)
        # with open("debug_output.bin", "wb") as f:
        #     f.write(binary)
        raw = decode(binary)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting raw thermal image: {e}")
        raise
    except Exception as e:
        print(f"Error decoding binary data: {e}")
        raise

    return raw, metadata


def raw2temp(raw, metadata):

    E = metadata["FLIR:Emissivity"]
    OD = metadata["FLIR:ObjectDistance"]
    RTemp = metadata["FLIR:ReflectedApparentTemperature"]
    ATemp = metadata["FLIR:AtmosphericTemperature"]
    IRWTemp = metadata["FLIR:IRWindowTemperature"]
    IRT = metadata["FLIR:IRWindowTransmission"]
    RH = metadata["FLIR:RelativeHumidity"]
    PR1 = metadata["FLIR:PlanckR1"]
    PB = metadata["FLIR:PlanckB"]
    PF = metadata["FLIR:PlanckF"]
    PO = metadata["FLIR:PlanckO"]
    PR2 = metadata["FLIR:PlanckR2"]
    ATA1 = float(metadata["FLIR:AtmosphericTransAlpha1"])
    ATA2 = float(metadata["FLIR:AtmosphericTransAlpha2"])
    ATB1 = float(metadata["FLIR:AtmosphericTransBeta1"])
    ATB2 = float(metadata["FLIR:AtmosphericTransBeta2"])
    ATX = metadata["FLIR:AtmosphericTransX"]

    emiss_wind = 1 - IRT
    refl_wind = 0
    h2o = (RH / 100) * exp(
        1.5587
        + 0.06939 * (ATemp)
        - 0.00027816 * (ATemp) ** 2
        + 0.00000068455 * (ATemp) ** 3
    )
    tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
        -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o))
    )
    tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
        -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o))
    )
    # Note: for this script, we assume the thermal window is at the mid-point (OD/2) between the source
    # and the camera sensor

    raw_refl1 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
    raw_refl1_attn = (1 - E) / E * raw_refl1

    raw_atm1 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
    raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1

    raw_wind = PR1 / (PR2 * (exp(PB / (IRWTemp + 273.15)) - PF)) - PO
    raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind

    raw_refl2 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
    raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2

    raw_atm2 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
    raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2

    raw_obj = (
        raw / E / tau1 / IRT / tau2
        - raw_atm1_attn
        - raw_atm2_attn
        - raw_wind_attn
        - raw_refl1_attn
        - raw_refl2_attn
    )

    temp_C = PB / log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15

    return temp_C


if __name__ == "__main__":

    from sys import argv
    import seaborn as sns
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def plot_thermal(frame):

        sns.set_style("ticks")
        fig = plt.figure()
        ax = plt.gca()
        plt.axis("off")
        im = plt.imshow(frame, cmap="hot")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel("Temperature ($^{\circ}$C)", fontsize=14)
        sns.despine()
        plt.show()

    reader = CSQReader(argv[1])

    frame = reader.next_frame()
    plot_thermal(frame)
