# SeaweedFS — single-node deploy

Runs on the dev server at `/mnt/hdd1/seaweedfs`. Blobs live on the 15TB
spinning disk; filer paths are stored in embedded leveldb2 (4.x default —
faster than Postgres on a single node and has no moving parts).

## Components

| service  | port | role |
|----------|------|------|
| master   | 9333 | cluster coordinator + volume allocator |
| volume   | 8080 | blob storage (writes to /mnt/hdd1/seaweedfs/data) |
| filer    | 8888 | path/S3 namespace (leveldb2 under /mnt/hdd1/seaweedfs/filery) |
| s3       | 8333 | S3-compatible gateway for agents / aws-cli / boto3 |

## Install

```sh
ssh user@192.168.0.1
cd /mnt/hdd1/seaweedfs/deploy      # docker-compose.yml + s3.json live here
docker compose up -d
```

First boot pulls one image (`chrislusf/seaweedfs:4.39`) and self-initialises.
Volume files appear under `/mnt/hdd1/seaweedfs/data/`; filer metadata under
`/mnt/hdd1/seaweedfs/filery/filerldb2/`.

## Use with aws-cli / boto3

```sh
ssh user@192.168.0.1
export PATH="$HOME/.local/bin:$PATH"      # awscli installed via uv

export AWS_ACCESS_KEY_ID=agent_key
export AWS_SECRET_ACCESS_KEY=agent_secret_dev_change_me
aws --endpoint-url http://127.0.0.1:8333 --region us-east-1 s3 mb s3://arxiv-papers
aws --endpoint-url http://127.0.0.1:8333 --region us-east-1 s3 cp /tmp/paper.pdf s3://arxiv-papers/2606.19348.pdf
aws --endpoint-url http://127.0.0.1:8333 --region us-east-1 s3 ls s3://arxiv-papers/
```

From a remote machine on the LAN, replace `127.0.0.1` with `192.168.0.1`.

## Identities (in `s3.json`)

- **admin** (`seaweedfs_admin` / `seaweedfs_admin_dev_secret_change_me`) — full access including Admin + bucket creation.
- **agent** (`agent_key` / `agent_secret_dev_change_me`) — Read/Write/List/Tagging on all buckets; no Admin.
- **anonymous** — Read only.

Rotate both secrets before exposing the gateway outside the LAN.

## Smoke test

```sh
bash smoke.sh
```

Verifies: master `/cluster/status`, volume server registration, S3 create-bucket
+ PUT + GET + LIST + DELETE roundtrip with both the admin and agent identities.

## Backups

A file-level snapshot of these two directories is a complete backup:

- `/mnt/hdd1/seaweedfs/data/`  — volume files (`.dat` + `.idx` per volume)
- `/mnt/hdd1/seaweedfs/filery/filerldb2/` — path/S3 namespace metadata

Both are append-only / write-ahead formats safe to copy while the cluster runs.

## HA path (when needed)

Single-master is fine while one disk is fine. To scale up:

1. Add 2 more `master` replicas with `-peers=master:9333,master2:9333,master3:9333`.
2. Add more `volume` servers, each with its own `-dir` (replication via
   `-defaultReplication` — e.g. `001` for one replica).
3. The filer and s3 gateway are stateless and can be replicated freely; at
   that point switch the filer store to Postgres for shared metadata.
