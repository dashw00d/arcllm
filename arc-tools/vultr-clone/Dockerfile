FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root/grabber

# System deps — mirrors Vultr cloud-init provisioner
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    gcc python3-dev \
    libxml2-dev libxslt1-dev \
    ndppd \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY grabber/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Grabber code (vendor brain + temporal orchestration)
COPY grabber/app/ app/
COPY grabber/vendor/ vendor/
COPY grabber/data/ data/
COPY grabber/temporal_worker.py grabber/temporal_activities.py grabber/temporal_workflows.py ./

CMD ["python3", "temporal_worker.py"]
