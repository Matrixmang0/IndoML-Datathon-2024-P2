# Use Python 3.11 base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /container

# Copy only the requirements file first to leverage Docker cache for dependencies
COPY requirements.txt /container/

# Create a virtual environment and install dependencies
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY ./data /container/data
COPY tuning_t5.py /container/
COPY entrypoint.sh /container/entrypoint.sh

# Set permissions for entrypoint.sh
RUN chmod +x /container/entrypoint.sh

# Set the entrypoint to display instructions
ENTRYPOINT ["/container/entrypoint.sh"]

# Keep the container open in an interactive shell
CMD ["/bin/bash"]

