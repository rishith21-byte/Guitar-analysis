import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QScrollArea,
    QSizePolicy,
    QMessageBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# =========================================================
# CONFIGURATION
# =========================================================

SAMPLE_RATE = 44100

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900

GRAPH_DURATION = 1.5
PITCH_HISTORY_LIMIT = 100

MIN_PITCH = 70.0
MAX_PITCH = 500.0

TUNE_TOLERANCE_CENTS = 5.0


VIRTUAL_STRINGS = {
    "A": {
        "note": "E2",
        "frequency": 82.41,
    },

    "S": {
        "note": "A2",
        "frequency": 110.00,
    },

    "D": {
        "note": "D3",
        "frequency": 146.83,
    },

    "F": {
        "note": "G3",
        "frequency": 196.00,
    },

    "G": {
        "note": "B3",
        "frequency": 246.94,
    },

    "H": {
        "note": "E4",
        "frequency": 329.63,
    }
}


# =========================================================
# CHORDS
# =========================================================

CHORDS = {
    "Q": {
        "name": "C",
        "frequencies": [
            261.63,
            329.63,
            392.00
        ]
    },

    "W": {
        "name": "G",
        "frequencies": [
            196.00,
            246.94,
            293.66
        ]
    },

    "E": {
        "name": "Am",
        "frequencies": [
            220.00,
            261.63,
            329.63
        ]
    },

    "R": {
        "name": "Em",
        "frequencies": [
            164.81,
            196.00,
            246.94
        ]
    }
}


# =========================================================
# GLOBAL VARIABLES
# =========================================================

current_signal = None
pitch_history = []


# =========================================================
# PITCH DETECTION
# =========================================================

def detect_pitch(audio, sample_rate):

    """
    Detect fundamental frequency using autocorrelation.
    """

    if audio is None:
        return None

    audio = np.asarray(
        audio,
        dtype=np.float64
    )

    if len(audio) < 100:
        return None

    # Remove DC component
    audio = audio - np.mean(audio)

    # Check signal strength
    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    if rms < 0.001:
        return None

    # Normalize
    maximum = np.max(
        np.abs(audio)
    )

    if maximum > 0:
        audio = audio / maximum

    # Apply Hann window
    audio = (
        audio *
        np.hanning(len(audio))
    )

    # Autocorrelation
    correlation = np.correlate(
        audio,
        audio,
        mode="full"
    )

    correlation = correlation[
        len(correlation) // 2:
    ]

    if len(correlation) < 2:
        return None

    # Search only in guitar frequency range
    min_lag = int(
        sample_rate /
        MAX_PITCH
    )

    max_lag = int(
        sample_rate /
        MIN_PITCH
    )

    max_lag = min(
        max_lag,
        len(correlation) - 1
    )

    if min_lag >= max_lag:
        return None

    search_region = correlation[
        min_lag:max_lag + 1
    ]

    if len(search_region) == 0:
        return None

    peak_index = np.argmax(
        search_region
    )

    lag = (
        peak_index +
        min_lag
    )

    if lag <= 0:
        return None

    frequency = (
        sample_rate /
        lag
    )

    if (
        frequency < MIN_PITCH
        or frequency > MAX_PITCH
    ):
        return None

    return float(frequency)


# =========================================================
# STABLE PITCH DETECTION
# =========================================================

def detect_best_pitch(
    audio,
    sample_rate
):

    """
    Detect pitch using multiple overlapping sections
    and return the median valid frequency.
    """

    if audio is None:
        return None

    audio = np.asarray(
        audio
    )

    if len(audio) == 0:
        return None

    chunk_size = int(
        0.75 *
        sample_rate
    )

    hop_size = int(
        0.25 *
        sample_rate
    )

    if len(audio) <= chunk_size:

        return detect_pitch(
            audio,
            sample_rate
        )

    detected_frequencies = []

    start = 0

    while (
        start + chunk_size
        <= len(audio)
    ):

        chunk = audio[
            start:
            start + chunk_size
        ]

        frequency = detect_pitch(
            chunk,
            sample_rate
        )

        if frequency is not None:
            detected_frequencies.append(
                frequency
            )

        start += hop_size

    if not detected_frequencies:
        return None

    return float(
        np.median(
            detected_frequencies
        )
    )


# =========================================================
# MATHEMATICAL GUITAR STRING SYNTHESIS
# =========================================================

def generate_guitar_string(
    frequency,
    duration=3.0
):

    """
    Generate a guitar-like plucked string mathematically.

    The sound contains:
    - Fundamental frequency
    - Multiple harmonics
    - Pluck attack
    - Exponential decay
    - Small high-frequency attack component

    No WAV files are used.
    """

    samples = int(
        SAMPLE_RATE *
        duration
    )

    t = (
        np.arange(samples) /
        SAMPLE_RATE
    )

    signal = np.zeros(
        samples,
        dtype=np.float64
    )

    harmonics = [
        (1, 1.00),
        (2, 0.45),
        (3, 0.25),
        (4, 0.15),
        (5, 0.09),
        (6, 0.06),
        (7, 0.04),
        (8, 0.025),
    ]

    for harmonic, amplitude in harmonics:

        signal += (
            amplitude *
            np.sin(
                2 *
                np.pi *
                frequency *
                harmonic *
                t
            )
        )

    # -----------------------------------------------------
    # Pluck attack
    # -----------------------------------------------------

    attack_time = 0.015

    attack_samples = max(
        1,
        int(
            attack_time *
            SAMPLE_RATE
        )
    )

    attack_envelope = np.ones(
        samples
    )

    attack_envelope[
        :attack_samples
    ] = np.linspace(
        0.0,
        1.0,
        attack_samples
    )

    # -----------------------------------------------------
    # String decay
    # -----------------------------------------------------

    decay_rate = 1.8

    decay_envelope = np.exp(
        -decay_rate *
        t
    )

    # -----------------------------------------------------
    # Combined envelope
    # -----------------------------------------------------

    envelope = (
        attack_envelope *
        decay_envelope
    )

    signal *= envelope

    # -----------------------------------------------------
    # Small pluck component
    # -----------------------------------------------------

    pluck_decay = np.exp(
        -35.0 * t
    )

    pluck_component = (
        0.08 *
        np.sin(
            2 *
            np.pi *
            frequency *
            8 *
            t
        ) *
        pluck_decay
    )

    signal += pluck_component

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    maximum = np.max(
        np.abs(signal)
    )

    if maximum > 0:
        signal /= maximum

    # Prevent excessive volume
    signal *= 0.55

    return signal.astype(
        np.float32
    )


# =========================================================
# MATHEMATICAL CHORD SYNTHESIS
# =========================================================

def generate_chord(
    frequencies,
    duration=2.5
):

    """
    Generate a mathematical guitar-style chord.
    """

    samples = int(
        SAMPLE_RATE *
        duration
    )

    t = (
        np.arange(samples) /
        SAMPLE_RATE
    )

    signal = np.zeros(
        samples,
        dtype=np.float64
    )

    # Generate each note with harmonics
    for frequency in frequencies:

        signal += (
            0.28 *
            np.sin(
                2 *
                np.pi *
                frequency *
                t
            )
        )

        signal += (
            0.12 *
            np.sin(
                2 *
                np.pi *
                frequency *
                2 *
                t
            )
        )

        signal += (
            0.06 *
            np.sin(
                2 *
                np.pi *
                frequency *
                3 *
                t
            )
        )

        signal += (
            0.03 *
            np.sin(
                2 *
                np.pi *
                frequency *
                4 *
                t
            )
        )

    # Chord decay
    envelope = np.exp(
        -1.4 * t
    )

    signal *= envelope

    # Normalize
    maximum = np.max(
        np.abs(signal)
    )

    if maximum > 0:
        signal /= maximum

    signal *= 0.5

    return signal.astype(
        np.float32
    )


# =========================================================
# MATPLOTLIB CANVAS
# =========================================================

class PlotCanvas(FigureCanvas):

    def __init__(
        self,
        parent=None
    ):

        self.figure, self.ax = plt.subplots(
            figsize=(10, 3.2)
        )

        self.figure.patch.set_facecolor(
            "#111111"
        )

        self.ax.set_facecolor(
            "#111111"
        )

        self.ax.tick_params(
            colors="white"
        )

        for spine in self.ax.spines.values():

            spine.set_color(
                "white"
            )

        super().__init__(
            self.figure
        )

        self.setParent(parent)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )


# =========================================================
# MAIN WINDOW
# =========================================================

class GuitarAnalyzer(QMainWindow):

    def __init__(self):

        super().__init__()

        self.current_signal = None
        self.pitch_history = []

        self.setWindowTitle(
            "🎸 Guitar Analyzer"
        )

        self.resize(
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )

        self.setMinimumSize(
            800,
            600
        )

        self.build_ui()

        self.apply_dark_theme()

        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )

        self.setFocus()


    # =====================================================
    # BUILD UI
    # =====================================================

    def build_ui(self):

        # -------------------------------------------------
        # SCROLLABLE DASHBOARD
        # -------------------------------------------------

        scroll_area = QScrollArea()

        scroll_area.setWidgetResizable(
            True
        )

        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.setCentralWidget(
            scroll_area
        )

        dashboard = QWidget()

        scroll_area.setWidget(
            dashboard
        )

        main_layout = QVBoxLayout(
            dashboard
        )

        main_layout.setContentsMargins(
            30,
            20,
            30,
            20
        )

        main_layout.setSpacing(
            12
        )


        # -------------------------------------------------
        # TITLE
        # -------------------------------------------------

        title_label = QLabel(
            "🎸 Guitar Analyzer"
        )

        title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        title_label.setStyleSheet(
            """
            QLabel {
                font-size: 26px;
                font-weight: bold;
                color: white;
            }
            """
        )

        main_layout.addWidget(
            title_label
        )


        subtitle_label = QLabel(
            "Guitar Tuning • Virtual Guitar • "
            "Pitch Detection • Signal Processing"
        )

        subtitle_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        subtitle_label.setStyleSheet(
            """
            QLabel {
                font-size: 11px;
                color: lightgray;
                padding-bottom: 10px;
            }
            """
        )

        main_layout.addWidget(
            subtitle_label
        )


        # -------------------------------------------------
        # ANALYZER FRAME
        # -------------------------------------------------

        analyzer_group = QGroupBox(
            "Guitar Analyzer"
        )

        analyzer_layout = QVBoxLayout(
            analyzer_group
        )

        self.note_label = QLabel(
            "--"
        )

        self.note_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.note_label.setStyleSheet(
            """
            QLabel {
                font-size: 42px;
                font-weight: bold;
                color: white;
            }
            """
        )

        analyzer_layout.addWidget(
            self.note_label
        )


        self.frequency_label = QLabel(
            "-- Hz"
        )

        self.frequency_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.frequency_label.setStyleSheet(
            """
            QLabel {
                font-size: 18px;
                color: white;
            }
            """
        )

        analyzer_layout.addWidget(
            self.frequency_label
        )


        self.target_label = QLabel(
            "Target: --"
        )

        self.target_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        analyzer_layout.addWidget(
            self.target_label
        )


        self.cents_label = QLabel(
            "+0.00 cents"
        )

        self.cents_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        analyzer_layout.addWidget(
            self.cents_label
        )


        self.tuning_status_label = QLabel(
            "WAITING"
        )

        self.tuning_status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.tuning_status_label.setStyleSheet(
            """
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: white;
            }
            """
        )

        analyzer_layout.addWidget(
            self.tuning_status_label
        )


        # -------------------------------------------------
        # TUNING METER
        # -------------------------------------------------

        self.meter_widget = QWidget()

        self.meter_widget.setMinimumHeight(
            50
        )

        meter_layout = QHBoxLayout(
            self.meter_widget
        )

        self.meter_left = QLabel(
            "FLAT"
        )

        self.meter_center = QLabel(
            "●"
        )

        self.meter_center.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.meter_right = QLabel(
            "SHARP"
        )

        meter_layout.addWidget(
            self.meter_left
        )

        meter_layout.addStretch()

        meter_layout.addWidget(
            self.meter_center
        )

        meter_layout.addStretch()

        meter_layout.addWidget(
            self.meter_right
        )

        analyzer_layout.addWidget(
            self.meter_widget
        )


        main_layout.addWidget(
            analyzer_group
        )


        # -------------------------------------------------
        # CONTROL BUTTONS
        # -------------------------------------------------

        control_frame = QWidget()

        control_layout = QHBoxLayout(
            control_frame
        )

        control_layout.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.stop_button = QPushButton(
            "⏹ Stop Sound"
        )

        self.stop_button.clicked.connect(
            self.stop_sound
        )

        control_layout.addWidget(
            self.stop_button
        )


        self.dark_button = QPushButton(
            "🌙 Dark Theme"
        )

        self.dark_button.clicked.connect(
            self.apply_dark_theme
        )

        control_layout.addWidget(
            self.dark_button
        )

        main_layout.addWidget(
            control_frame
        )


        # -------------------------------------------------
        # VIRTUAL GUITAR
        # -------------------------------------------------

        virtual_group = QGroupBox(
            "Virtual Guitar"
        )

        virtual_layout = QVBoxLayout(
            virtual_group
        )


        self.virtual_status_label = QLabel(
            "Select a guitar string"
        )

        self.virtual_status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        virtual_layout.addWidget(
            self.virtual_status_label
        )


        string_button_frame = QWidget()

        string_layout = QGridLayout(
            string_button_frame
        )

        string_layout.setSpacing(
            8
        )


        for column, (
            key,
            data
        ) in enumerate(
            VIRTUAL_STRINGS.items()
        ):

            button = QPushButton(
                f"{key}\n"
                f"{data['note']}\n"
                f"{data['frequency']:.2f} Hz"
            )

            button.setMinimumHeight(
                75
            )

            button.setMinimumWidth(
                120
            )

            button.clicked.connect(
                lambda checked=False,
                k=key:
                self.play_virtual_string(k)
            )

            string_layout.addWidget(
                button,
                0,
                column
            )


        virtual_layout.addWidget(
            string_button_frame
        )

        main_layout.addWidget(
            virtual_group
        )


        # -------------------------------------------------
        # CHORDS
        # -------------------------------------------------

        chord_group = QGroupBox(
            "Chords"
        )

        chord_layout = QHBoxLayout(
            chord_group
        )

        for key, data in CHORDS.items():

            chord_button = QPushButton(
                f"{key}\n"
                f"{data['name']}"
            )

            chord_button.setMinimumHeight(
                70
            )

            chord_button.setMinimumWidth(
                120
            )

            chord_button.clicked.connect(
                lambda checked=False,
                k=key:
                self.play_chord(k)
            )

            chord_layout.addWidget(
                chord_button
            )

        main_layout.addWidget(
            chord_group
        )


        # -------------------------------------------------
        # GRAPH FRAME
        # -------------------------------------------------

        graph_group = QGroupBox(
            "Signal Analysis"
        )

        graph_layout = QVBoxLayout(
            graph_group
        )


        # Waveform
        waveform_title = QLabel(
            "Waveform"
        )

        waveform_title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        graph_layout.addWidget(
            waveform_title
        )


        self.waveform_canvas = PlotCanvas()

        self.waveform_canvas.setMinimumHeight(
            300
        )

        graph_layout.addWidget(
            self.waveform_canvas
        )


        # Spectrum
        spectrum_title = QLabel(
            "Frequency Spectrum"
        )

        spectrum_title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        graph_layout.addWidget(
            spectrum_title
        )


        self.spectrum_canvas = PlotCanvas()

        self.spectrum_canvas.setMinimumHeight(
            300
        )

        graph_layout.addWidget(
            self.spectrum_canvas
        )


        # Pitch history
        history_title = QLabel(
            "Pitch History"
        )

        history_title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        graph_layout.addWidget(
            history_title
        )


        self.pitch_history_canvas = PlotCanvas()

        self.pitch_history_canvas.setMinimumHeight(
            300
        )

        graph_layout.addWidget(
            self.pitch_history_canvas
        )


        main_layout.addWidget(
            graph_group
        )


        # -------------------------------------------------
        # STANDARD TUNING TABLE
        # -------------------------------------------------

        tuning_group = QGroupBox(
            "Standard Guitar Tuning"
        )

        tuning_layout = QGridLayout(
            tuning_group
        )


        headers = [
            "String",
            "Note",
            "Frequency"
        ]


        for column, header in enumerate(
            headers
        ):

            label = QLabel(
                header
            )

            label.setAlignment(
                Qt.AlignmentFlag.AlignCenter
            )

            label.setStyleSheet(
                """
                QLabel {
                    font-weight: bold;
                    padding: 8px;
                }
                """
            )

            tuning_layout.addWidget(
                label,
                0,
                column
            )


        standard_strings = [
            (
                "6th String",
                "E2",
                "82.41 Hz"
            ),

            (
                "5th String",
                "A2",
                "110.00 Hz"
            ),

            (
                "4th String",
                "D3",
                "146.83 Hz"
            ),

            (
                "3rd String",
                "G3",
                "196.00 Hz"
            ),

            (
                "2nd String",
                "B3",
                "246.94 Hz"
            ),

            (
                "1st String",
                "E4",
                "329.63 Hz"
            ),
        ]


        for row, values in enumerate(
            standard_strings,
            start=1
        ):

            for column, value in enumerate(
                values
            ):

                label = QLabel(
                    value
                )

                label.setAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )

                label.setStyleSheet(
                    """
                    QLabel {
                        padding: 7px;
                        border: 1px solid #444444;
                    }
                    """
                )

                tuning_layout.addWidget(
                    label,
                    row,
                    column
                )


        main_layout.addWidget(
            tuning_group
        )


        # -------------------------------------------------
        # FOOTER
        # -------------------------------------------------

        footer_label = QLabel(
            "Guitar Analyzer • "
            "Mathematical Guitar Synthesis • "
            "Pitch Detection • Signal Processing"
        )

        footer_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        footer_label.setStyleSheet(
            """
            QLabel {
                font-size: 10px;
                color: gray;
                padding: 15px;
            }
            """
        )

        main_layout.addWidget(
            footer_label
        )


        main_layout.addStretch()


    # =====================================================
    # UPDATE ANALYZER
    # =====================================================

    def update_analyzer(
        self,
        detected_note,
        detected_frequency,
        target_note,
        target_frequency,
        signal=None
    ):

        if detected_frequency is not None:

            self.frequency_label.setText(
                f"{detected_frequency:.2f} Hz"
            )

        else:

            self.frequency_label.setText(
                "-- Hz"
            )


        self.note_label.setText(
            detected_note
        )


        self.target_label.setText(
            f"Target: {target_note} "
            f"({target_frequency:.2f} Hz)"
        )


        if (
            detected_frequency is not None
            and detected_frequency > 0
            and target_frequency > 0
        ):

            cents = (
                1200 *
                np.log2(
                    detected_frequency /
                    target_frequency
                )
            )

        else:

            cents = 0.0


        self.cents_label.setText(
            f"{cents:+.2f} cents"
        )


        if abs(cents) <= TUNE_TOLERANCE_CENTS:

            self.tuning_status_label.setText(
                "IN TUNE"
            )

        elif cents < 0:

            self.tuning_status_label.setText(
                "FLAT"
            )

        else:

            self.tuning_status_label.setText(
                "SHARP"
            )


        # -------------------------------------------------
        # Pitch history
        # -------------------------------------------------

        if detected_frequency is not None:

            self.pitch_history.append(
                detected_frequency
            )

            if (
                len(self.pitch_history)
                > PITCH_HISTORY_LIMIT
            ):

                self.pitch_history.pop(0)


        # -------------------------------------------------
        # Graphs
        # -------------------------------------------------

        self.update_waveform(
            signal
        )

        self.update_spectrum(
            signal
        )

        self.update_pitch_history()


    # =====================================================
    # WAVEFORM
    # =====================================================

    def update_waveform(
        self,
        signal
    ):

        ax = self.waveform_canvas.ax

        ax.clear()

        ax.set_title(
            "Waveform",
            color="white"
        )

        ax.set_xlabel(
            "Time (seconds)",
            color="white"
        )

        ax.set_ylabel(
            "Amplitude",
            color="white"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.tick_params(
            colors="white"
        )

        for spine in ax.spines.values():

            spine.set_color(
                "white"
            )


        if signal is None:

            self.waveform_canvas.draw()

            return


        samples_to_show = min(
            len(signal),
            int(
                SAMPLE_RATE *
                GRAPH_DURATION
            )
        )


        waveform = signal[
            :samples_to_show
        ]


        time = (
            np.arange(
                len(waveform)
            ) /
            SAMPLE_RATE
        )


        ax.plot(
            time,
            waveform
        )

        self.waveform_canvas.figure.tight_layout()

        self.waveform_canvas.draw()


    # =====================================================
    # SPECTRUM
    # =====================================================

    def update_spectrum(
        self,
        signal
    ):

        ax = self.spectrum_canvas.ax

        ax.clear()

        ax.set_title(
            "Frequency Spectrum",
            color="white"
        )

        ax.set_xlabel(
            "Frequency (Hz)",
            color="white"
        )

        ax.set_ylabel(
            "Magnitude",
            color="white"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.tick_params(
            colors="white"
        )

        for spine in ax.spines.values():

            spine.set_color(
                "white"
            )


        if signal is None:

            self.spectrum_canvas.draw()

            return


        samples_to_use = min(
            len(signal),
            SAMPLE_RATE
        )


        audio = signal[
            :samples_to_use
        ]


        if len(audio) == 0:

            self.spectrum_canvas.draw()

            return


        window = np.hanning(
            len(audio)
        )


        audio = (
            audio *
            window
        )


        fft = np.fft.rfft(
            audio
        )


        frequencies = np.fft.rfftfreq(
            len(audio),
            1 / SAMPLE_RATE
        )


        magnitude = np.abs(
            fft
        )


        ax.plot(
            frequencies,
            magnitude
        )


        ax.set_xlim(
            0,
            1000
        )


        self.spectrum_canvas.figure.tight_layout()

        self.spectrum_canvas.draw()


    # =====================================================
    # PITCH HISTORY
    # =====================================================

    def update_pitch_history(self):

        ax = self.pitch_history_canvas.ax

        ax.clear()

        ax.set_title(
            "Pitch History",
            color="white"
        )

        ax.set_xlabel(
            "Sample",
            color="white"
        )

        ax.set_ylabel(
            "Frequency (Hz)",
            color="white"
        )

        ax.grid(
            True,
            alpha=0.3
        )

        ax.tick_params(
            colors="white"
        )

        for spine in ax.spines.values():

            spine.set_color(
                "white"
            )


        if self.pitch_history:

            ax.plot(
                self.pitch_history
            )


        self.pitch_history_canvas.figure.tight_layout()

        self.pitch_history_canvas.draw()


    # =====================================================
    # PLAY VIRTUAL STRING
    # =====================================================

    def play_virtual_string(
        self,
        key
    ):

        global current_signal

        if key not in VIRTUAL_STRINGS:
            return


        data = VIRTUAL_STRINGS[key]

        target_note = data["note"]

        target_frequency = data["frequency"]


        # -------------------------------------------------
        # GENERATE STRING MATHEMATICALLY
        # -------------------------------------------------

        signal = generate_guitar_string(
            target_frequency,
            duration=3.0
        )


        current_signal = signal

        self.current_signal = signal


        # -------------------------------------------------
        # PLAY SOUND
        # -------------------------------------------------

        try:

            sd.stop()

            sd.play(
                signal,
                SAMPLE_RATE
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Virtual Guitar Error",
                (
                    "Could not play the "
                    "virtual guitar sound.\n\n"
                    f"Error:\n{error}"
                )
            )

            return


        # -------------------------------------------------
        # ANALYZE GENERATED SIGNAL
        # -------------------------------------------------
        #
        # This uses pitch detection only.
        # No ML or WAV analysis is used.
        # -------------------------------------------------

        detected_frequency = (
            detect_best_pitch(
                signal,
                SAMPLE_RATE
            )
        )


        if detected_frequency is not None:

            detected_note = (
                self.frequency_to_note(
                    detected_frequency
                )
            )

        else:

            detected_frequency = (
                target_frequency
            )

            detected_note = (
                target_note
            )


        # -------------------------------------------------
        # UPDATE ANALYZER
        # -------------------------------------------------

        self.update_analyzer(
            detected_note=detected_note,
            detected_frequency=detected_frequency,
            target_note=target_note,
            target_frequency=target_frequency,
            signal=signal
        )


        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        self.virtual_status_label.setText(
            (
                f"Selected String: {key}   |   "
                f"{target_note}   |   "
                f"{target_frequency:.2f} Hz   |   "
                "Mathematical Guitar Synthesis"
            )
        )


        print(
            f"Playing {key}: "
            f"{target_note} "
            f"({target_frequency:.2f} Hz)"
        )


    # =====================================================
    # PLAY CHORD
    # =====================================================

    def play_chord(
        self,
        key
    ):

        if key not in CHORDS:
            return


        chord = CHORDS[key]


        signal = generate_chord(
            chord["frequencies"]
        )


        try:

            sd.stop()

            sd.play(
                signal,
                SAMPLE_RATE
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Chord Playback Error",
                (
                    "Could not play the chord.\n\n"
                    f"Error:\n{error}"
                )
            )

            return


        global current_signal

        current_signal = signal

        self.current_signal = signal


        # Display chord waveform

        self.update_waveform(
            signal
        )

        self.update_spectrum(
            signal
        )

        self.update_pitch_history()


        self.virtual_status_label.setText(
            (
                f"Chord: "
                f"{chord['name']}   |   "
                "Mathematical Synthesis"
            )
        )


    # =====================================================
    # STOP SOUND
    # =====================================================

    def stop_sound(self):

        try:

            sd.stop()

        except Exception:

            pass


    # =====================================================
    # DARK THEME
    # =====================================================

    def apply_dark_theme(self):

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #111111;
            }

            QWidget {
                background-color: #111111;
                color: white;
            }

            QGroupBox {
                border: 1px solid #444444;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px;
                font-size: 13px;
                font-weight: bold;
                color: white;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }

            QPushButton {
                background-color: #222222;
                color: white;
                border: 1px solid #555555;
                border-radius: 7px;
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #333333;
            }

            QPushButton:pressed {
                background-color: #444444;
            }

            QScrollArea {
                border: none;
                background-color: #111111;
            }

            QScrollBar:vertical {
                background: #1b1b1b;
                width: 13px;
                margin: 0px;
            }

            QScrollBar::handle:vertical {
                background: #555555;
                min-height: 35px;
                border-radius: 6px;
            }

            QScrollBar::handle:vertical:hover {
                background: #777777;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )


    # =====================================================
    # KEYBOARD CONTROLS
    # =====================================================

    def keyPressEvent(
        self,
        event
    ):

        key = event.text().upper()

        if key in VIRTUAL_STRINGS:

            self.play_virtual_string(
                key
            )

        elif key in CHORDS:

            self.play_chord(
                key
            )

        else:

            super().keyPressEvent(
                event
            )


    # =====================================================
    # CLOSE APPLICATION
    # =====================================================

    def closeEvent(
        self,
        event
    ):

        try:

            sd.stop()

        except Exception:

            pass

        event.accept()


    # =====================================================
    # NOTE CONVERSION
    # =====================================================

    @staticmethod
    def frequency_to_note(
        frequency
    ):

        if frequency <= 0:
            return "--"

        note_names = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B"
        ]

        midi = round(
            69 +
            12 *
            np.log2(
                frequency / 440.0
            )
        )

        note_index = midi % 12

        octave = (
            midi // 12
        ) - 1

        return (
            f"{note_names[note_index]}"
            f"{octave}"
        )


# =========================================================
# START APPLICATION
# =========================================================

def main():

    app = QApplication(
        sys.argv
    )

    window = GuitarAnalyzer()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()
