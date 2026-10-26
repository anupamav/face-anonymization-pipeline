from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "images"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PAIR_ID = 1
# Add rgb_1.npy + depth_1.npy etc. to images/. Alternatively put your ZIP there.
RGB_FILENAME = None  # None -> rgb_{PAIR_ID}.npy; also supports .png/.jpg
DEPTH_FILENAME = None  # None -> depth_{PAIR_ID}.npy; also supports raw PNG/TIFF
INPUT_ZIP = None  # e.g. "doll_npy (1).zip"; resolved under images/
FX, FY, CX, CY = 649.872, 649.872, 362.120112, 653.496832
CALIBRATION_SHAPE = (1280, 720)  # H,W; set None if using other verified calibration
DEPTH_SCALE = 1.0
MIN_DEPTH_M, MAX_DEPTH_M = 0.1, 8.0
REGISTERED_TO_RGB = True
PORTRAIT_CALIBRATION_CONFIRMED = True
PERSON_INDEX = None
SAM_MASK_INDEX = None  # None: smallest SAM mask containing image centre
MANUAL_CHILD_POLYGON = None
GRAVITY_CAMERA = None
FORCE_MANUAL = False  # True: draw face rectangle even if automatic masking would succeed
EXPORT_REVIEWED = False
