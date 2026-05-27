

set -euxo pipefail
# set -e  → exit immediately on any error
# set -u  → treat unset variables as error
# set -x  → print each command before running (visible in EMR bootstrap logs)
# set -o pipefail → catch errors inside pipes too



# ─── CONFIG ─────────────────────────────────────────────────
S3_BUCKET="s3://recommendation-system-1149"
emr_requirements_PATH="$S3_BUCKET/code/emr_requirements.txt"
VENV_PATH="/home/hadoop/pipeline_venv"
PYTHON_BIN="/usr/bin/python3"   # match your EMR release (EMR 6.x ships 3.9)

# ─── STEP 1: Upgrade pip + install virtualenv ────────────────
echo "[$(date)] Upgrading pip and installing virtualenv..."
# sudo $PYTHON_BIN -m pip install --upgrade pip --quiet
sudo $PYTHON_BIN -m pip install virtualenv --quiet

# ─── STEP 2: Create the venv ────────────────────────────────
echo "[$(date)] Creating virtualenv at $VENV_PATH..."
sudo $PYTHON_BIN -m virtualenv "$VENV_PATH"

# ─── STEP 3: Download emr_requirements.txt from S3 ──────────────
echo "[$(date)] Downloading emr_requirements from S3..."
aws s3 cp "$emr_requirements_PATH" /tmp/emr_requirements.txt

# ─── STEP 4: Install all packages INTO the venv ─────────────
echo "[$(date)] Installing packages into venv..."
sudo "$VENV_PATH/bin/pip" install --upgrade pip --quiet
sudo "$VENV_PATH/bin/pip" install -r /tmp/emr_requirements.txt \
    --no-cache-dir \
    --quiet

# ─── STEP 5: Set ownership so hadoop user can use it ────────
# CRITICAL: EMR runs Spark executors as the 'hadoop' user, not root
sudo chown -R hadoop:hadoop "$VENV_PATH"

# ─── STEP 6: Tell Spark to use this venv's Python ───────────
# These env vars are read by YARN when launching executor processes
# They must be set BEFORE Spark starts, which is why bootstrap is the right place

SPARK_ENV_FILE="/etc/spark/conf/spark-env.sh"

echo "export PYSPARK_PYTHON=$VENV_PATH/bin/python" \
    | sudo tee -a "$SPARK_ENV_FILE"

echo "export PYSPARK_DRIVER_PYTHON=$VENV_PATH/bin/python" \
    | sudo tee -a "$SPARK_ENV_FILE"

# Also export for the shell session (for interactive use / EMR Studio)
echo "export PYSPARK_PYTHON=$VENV_PATH/bin/python" \
    | sudo tee -a /etc/environment
echo "export PYSPARK_DRIVER_PYTHON=$VENV_PATH/bin/python" \
    | sudo tee -a /etc/environment

# ─── STEP 7: Verify installation ────────────────────────────
echo "[$(date)] Verifying install..."
"$VENV_PATH/bin/python" -c "
import pyarrow, boto3
print('pyarrow:', pyarrow.__version__)
print('All packages verified.')
"

echo "[$(date)] Bootstrap complete on $(hostname)"