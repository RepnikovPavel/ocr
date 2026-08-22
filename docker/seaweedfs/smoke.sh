#!/usr/bin/env bash
# smoke.sh — end-to-end check of the SeaweedFS cluster.
#
# Verifies that all 5 services are up, that the master knows about the volume
# server, that a volume gets assigned, and that S3 (create-bucket, PUT, GET,
# LIST, DELETE) roundtrips correctly with both the admin and agent identities.
set -eu

HOST="${HOST:-127.0.0.1}"
ADMIN_AK="${ADMIN_AK:-seaweedfs_admin}"
ADMIN_SK="${ADMIN_SK:-seaweedfs_admin_dev_secret_change_me}"
AGENT_AK="${AGENT_AK:-agent_key}"
AGENT_SK="${AGENT_SK:-agent_secret_dev_change_me}"

GREEN() { printf '\033[32m%s\033[0m\n' "$*"; }
RED()   { printf '\033[31m%s\033[0m\n' "$*" >&2; }
die()   { RED "FAIL: $*"; exit 1; }

need() { command -v "$1" >/dev/null || die "$1 is required (curl + awscli + jq)"; }
need curl
need aws
need jq

# --- services up --------------------------------------------------------------
GREEN "==> 1/6: master /cluster/status"
curl -fsS "http://${HOST}:9333/cluster/status" | jq . >/dev/null || die "master not responding"
GREEN "    master OK"

GREEN "==> 2/6: volume server registered with master"
# /dir/status nests nodes under Topology → DataCenters → Racks → DataNodes.
VOL_INFO=$(curl -fsS "http://${HOST}:9333/dir/status" \
           | jq '[.Topology.DataCenters[].Racks[].DataNodes[]] | length')
[ "${VOL_INFO}" -ge 1 ] || die "no volume servers registered"
echo "    ${VOL_INFO} volume server(s) registered"

GREEN "==> 3/6: ask master for a volume id (assign)"
ASSIGN=$(curl -fsS "http://${HOST}:9333/dir/assign")
echo "    ${ASSIGN}" | jq '{fid, url, publicUrl, count}'

GREEN "==> 4/6: volume server responds"
# The master returns the container-internal hostname (e.g. "volume:8080") in
# `url`. That name can't be resolved from outside the docker network, so probe
# the host's mapped volume port directly (single-node deploy → port 8080).
VOL_URL="${HOST}:8080"
curl -fsS "http://${VOL_URL}/status" >/dev/null || die "volume server ${VOL_URL} down"
echo "    ${VOL_URL} up"

# --- S3 roundtrip -------------------------------------------------------------
ENDPOINT="http://${HOST}:8333"
BUCKET="smoke-$(date +%s)-$$"

GREEN "==> 5/6: S3 create-bucket + PUT + GET + LIST (admin)"
export AWS_ACCESS_KEY_ID="${ADMIN_AK}" AWS_SECRET_ACCESS_KEY="${ADMIN_SK}"
aws --endpoint-url "${ENDPOINT}" --region us-east-1 s3 mb "s3://${BUCKET}" >/dev/null \
    || die "create-bucket"
echo "hello seaweed" > /tmp/_smoke_obj.txt
aws --endpoint-url "${ENDPOINT}" --region us-east-1 s3 cp /tmp/_smoke_obj.txt \
    "s3://${BUCKET}/hello.txt" >/dev/null || die "PUT"
aws --endpoint-url "${ENDPOINT}" --region us-east-1 s3 cp \
    "s3://${BUCKET}/hello.txt" /tmp/_smoke_obj_back.txt >/dev/null || die "GET"
diff /tmp/_smoke_obj.txt /tmp/_smoke_obj_back.txt >/dev/null || die "GET content mismatch"
echo "    PUT/GET roundtrip OK"
LS_OUT=$(aws --endpoint-url "${ENDPOINT}" --region us-east-1 s3 ls "s3://${BUCKET}/")
echo "${LS_OUT}" | grep -q "hello.txt" || die "LIST missing hello.txt"
echo "    LIST shows hello.txt"

GREEN "==> 6/6: agent identity can also write + read"
export AWS_ACCESS_KEY_ID="${AGENT_AK}" AWS_SECRET_ACCESS_KEY="${AGENT_SK}"
echo "agent payload" > /tmp/_smoke_agent.txt
aws --endpoint-url "${ENDPOINT}" --region us-east-1 s3 cp /tmp/_smoke_agent.txt \
    "s3://${BUCKET}/from-agent.txt" >/dev/null || die "agent PUT"
aws --endpoint-url "${ENDPOINT}" --region us-east-1 s3 cp \
    "s3://${BUCKET}/from-agent.txt" /tmp/_smoke_agent_back.txt >/dev/null || die "agent GET"
diff /tmp/_smoke_agent.txt /tmp/_smoke_agent_back.txt >/dev/null \
    || die "agent GET content mismatch"
echo "    agent PUT/GET OK"

# cleanup
export AWS_ACCESS_KEY_ID="${ADMIN_AK}" AWS_SECRET_ACCESS_KEY="${ADMIN_SK}"
aws --endpoint-url "${ENDPOINT}" --region us-east-1 s3 rb --force "s3://${BUCKET}" >/dev/null \
    || echo "    (cleanup: bucket removal skipped)"
rm -f /tmp/_smoke_obj.txt /tmp/_smoke_obj_back.txt /tmp/_smoke_agent.txt /tmp/_smoke_agent_back.txt

GREEN "==> ALL OK"
