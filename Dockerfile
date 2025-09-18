FROM soarsmu/bugsinpy:latest

# Add basic tools (assuming Ubuntu/Debian base)
RUN apt-get update && \
    apt-get install -y bash coreutils python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set a working shell
CMD ["/bin/bash"]
