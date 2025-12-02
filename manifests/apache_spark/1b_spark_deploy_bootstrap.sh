#!/bin/bash
set -euxo pipefail

# ============================================================================== 
# ## Config
# ============================================================================== 
# Weka & Shared Storage
default_weka_cluster_address="10.1.3.66"
weka_mount_point="/mnt/weka"

# S3 Software Location
s3_bucket="weka-alan-sparktest-data-bucket"
WORKING_BUCKET="s3://${s3_bucket}"
s3_path="${WORKING_BUCKET}/sw"

# Spark
spark_version="3.5.5"
spark_install_dir="/opt/spark"
spark_pkg="spark-${spark_version}-bin-hadoop3"
spark_tar="${spark_pkg}.tgz"
spark_url="https://archive.apache.org/dist/spark/spark-${spark_version}/${spark_tar}"
spark_tmp_path="/tmp/${spark_tar}"
export SPARK_HOME="${spark_install_dir}/spark"
export PATH="$SPARK_HOME/bin:$PATH"

# ZooKeeper
zk_version="3.9.4"
zk_install_dir="/opt/zookeeper"
zk_pkg="apache-zookeeper-${zk_version}-bin"
zk_tar="${zk_pkg}.tar.gz"
zk_url="https://archive.apache.org/dist/zookeeper/zookeeper-${zk_version}/${zk_tar}"
zk_tmp_path="/tmp/${zk_tar}"
zk_data_dir="/var/lib/zookeeper"

# Cluster size
readonly DESIRED_ASG_COUNT=$(( $1 > 0 ? $1 : 3 ))

# ============================================================================== 
# ## 1. Prerequisites and Mounts
# ============================================================================== 
if command -v dnf &> /dev/null; then
    dnf install -y java-11-amazon-corretto-devel jq aws-cli
else
    amazon-linux-extras install -y java-11-openjdk-devel
    yum install -y jq aws-cli
fi

if findmnt --noheadings --target "${weka_mount_point}" | grep -q "wekafs"; then
    echo "WekaFS is already mounted. Discovering source address..."
    weka_cluster_address=$(findmnt -t wekafs -no SOURCE | head -n 1 | awk -F'/' '{print $1}')
    echo "Discovered Weka cluster address: ${weka_cluster_address}"
else
    echo "WekaFS not mounted. Using default address for installation."
    weka_cluster_address="${default_weka_cluster_address}"
    curl -L http://${weka_cluster_address}:14000/dist/v1/install | sh
    mkdir -p "${weka_mount_point}"
    mount -t wekafs -o num_cores=1 "${weka_cluster_address}/default" "${weka_mount_point}"
fi

# ============================================================================== 
# ## 2. Download and Install Software
# ============================================================================== 
mkdir -p "${spark_install_dir}"

if ! aws s3 cp "${s3_path}/${spark_tar}" "${spark_tmp_path}"; then
    echo "Spark installer not found in S3. Downloading from public archive..."
    curl -L -o "${spark_tmp_path}" "${spark_url}"
fi

tar -xzf "${spark_tmp_path}" -C "${spark_install_dir}"
ln -sfn "${spark_install_dir}/${spark_pkg}" "${SPARK_HOME}"

mkdir -p "${zk_install_dir}"

if ! aws s3 cp "${s3_path}/${zk_tar}" "${zk_tmp_path}"; then
    echo "ZooKeeper installer not found in S3. Downloading from public archive..."
    curl -L -o "${zk_tmp_path}" "${zk_url}"
fi

tar -xzf "${zk_tmp_path}" -C "${zk_install_dir}"
ln -sfn "${zk_install_dir}/${zk_pkg}" "${zk_install_dir}/zookeeper"

curl -L https://www.scala-sbt.org/sbt-rpm.repo > /etc/yum.repos.d/sbt-rpm.repo
yum install -y nc git sbt htop python3-devel
python3 -m ensurepip --upgrade
wget -P /tmp https://dev.yorhel.nl/download/ncdu-2.9.1-linux-x86_64.tar.gz
tar -zxvf /tmp/ncdu-2.9.1-linux-x86_64.tar.gz
mv /tmp/ncdu /usr/local/bin/
pip3 install psutil
pip3 install pyspark

# ============================================================================== 
# ## 3. Discover Peers and Configure ZooKeeper
# ============================================================================== 
echo "Discovering peers in the Auto Scaling Group..."
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: ${TOKEN}" http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: ${TOKEN}" http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r .region)
ASG_NAME=$(aws ec2 describe-tags --region "${REGION}" --filters "Name=resource-id,Values=${INSTANCE_ID}" "Name=key,Values=aws:autoscaling:groupName" --query "Tags[0].Value" --output text)

while true; do
    INSTANCE_INFO=$(aws ec2 describe-instances --region "${REGION}" --filters "Name=tag:aws:autoscaling:groupName,Values=${ASG_NAME}" "Name=instance-state-name,Values=running" --query "Reservations[].Instances[].[InstanceId,PrivateIpAddress]" --output json)
    INSTANCE_COUNT=$(echo "${INSTANCE_INFO}" | jq 'length')
    if [ "${INSTANCE_COUNT}" -ge "${DESIRED_ASG_COUNT}" ]; then break; fi
    echo "Found ${INSTANCE_COUNT}/${DESIRED_ASG_COUNT} instances. Waiting..."
    sleep 10
done

SORTED_INSTANCES=$(echo "${INSTANCE_INFO}" | jq -s '.[0] | sort_by(.[0])')
MASTER_IPS=()
ZK_HOSTS=""
MY_ID=0

ZOO_CFG_PATH="${zk_install_dir}/zookeeper/conf/zoo.cfg"

cat > "${ZOO_CFG_PATH}" <<EOF
tickTime=2000
dataDir=${zk_data_dir}
clientPort=2181
initLimit=5
syncLimit=2
EOF

for i in $(seq 0 $((${DESIRED_ASG_COUNT} - 1))); do
    id=$(echo "${SORTED_INSTANCES}" | jq -r ".[${i}][0]")
    ip=$(echo "${SORTED_INSTANCES}" | jq -r ".[${i}][1]")
    MASTER_IPS+=("${ip}")
    echo "server.$((${i} + 1))=${ip}:2888:3888" >> "${ZOO_CFG_PATH}"
    ZK_HOSTS+="${ip}:2181,"
    if [ "${id}" == "${INSTANCE_ID}" ]; then
        MY_ID=$((${i} + 1))
    fi
done

ZK_HOSTS=${ZK_HOSTS%,}
mkdir -p "${zk_data_dir}"
echo "${MY_ID}" > "${zk_data_dir}/myid"

# ============================================================================== 
# ## 4. Start ZooKeeper Service
# ============================================================================== 
echo "Starting ZooKeeper (My ID: ${MY_ID})..."
nohup "${zk_install_dir}/zookeeper/bin/zkServer.sh" start > /var/log/zookeeper_start.log 2>&1 &
sleep 15

# ============================================================================== 
# ## 5. Configure and Start Spark Cluster
# ============================================================================== 
echo "Configuring Spark for High Availability..."

MASTER_URLS=""
for ip in "${MASTER_IPS[@]}"; do
  MASTER_URLS+="${ip}:7077,"
done
MASTER_URLS=${MASTER_URLS%,}
SPARK_MASTER_URL="spark://${MASTER_URLS}"

cat > "${SPARK_HOME}/conf/spark-env.sh" <<EOF
#!/usr/bin/env bash
export SPARK_DAEMON_JAVA_OPTS="-Dspark.deploy.recoveryMode=ZOOKEEPER -Dspark.deploy.zookeeper.url=${ZK_HOSTS} -Dspark.deploy.zookeeper.dir=/spark"
EOF

# Enable S3 support in spark-defaults.conf
SPARK_DEFAULTS="$SPARK_HOME/conf/spark-defaults.conf"
if [ ! -f "$SPARK_DEFAULTS" ]; then
  cp "$SPARK_HOME/conf/spark-defaults.conf.template" "$SPARK_DEFAULTS"
fi

# Ensure warehouse directory exists for Hive metastore
WAREHOUSE_DIR="/mnt/weka/warehouse"
mkdir -p "$WAREHOUSE_DIR"

cat >> "$SPARK_DEFAULTS" <<EOF
spark.hadoop.fs.s3a.impl org.apache.hadoop.fs.s3a.S3AFileSystem
spark.hadoop.fs.AbstractFileSystem.s3a.impl org.apache.hadoop.fs.s3a.S3A
spark.hadoop.fs.s3a.statistics=all
spark.hadoop.fs.s3a.aws.credentials.provider com.amazonaws.auth.DefaultAWSCredentialsProviderChain
spark.sql.warehouse.dir $WAREHOUSE_DIR
javax.jdo.option.ConnectionURL jdbc:derby:/mnt/weka/metastore_db;create=true
EOF

# Download Hadoop AWS + AWS SDK jars
cd "$SPARK_HOME/jars"
if [ ! -f "hadoop-aws-3.3.4.jar" ]; then
  wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar
fi
if [ ! -f "aws-java-sdk-bundle-1.12.262.jar" ]; then
  wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar
fi

# Start Spark master and worker

echo "Starting Spark Master (HA; this node may be ACTIVE or STANDBY)..."
nohup "${SPARK_HOME}/sbin/start-master.sh" > /var/log/spark-master.log 2>&1 &

echo "Starting Spark Worker, targeting: ${SPARK_MASTER_URL}"
nohup "${SPARK_HOME}/sbin/start-worker.sh" "${SPARK_MASTER_URL}" > /var/log/spark-worker.log 2>&1 &

aws s3 sync ${WORKING_BUCKET}/scripts/ /mnt/weka/scripts/ && chmod a+x /mnt/weka/scripts/*
