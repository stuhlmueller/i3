import cloud
import os


def upload_data():
  """Upload data to picloud buckets."""
  local_data_path = os.path.join(os.path.dirname(__file__), "../data/")
  for root, _, files in os.walk(local_data_path):
    for file in files:
      if not file.startswith("."):
        source_file = os.path.join(root, file)
        target_file = os.path.join(os.path.relpath(root, local_data_path), file)
        print "Uploading {} to {}".format(source_file, target_file)
        cloud.bucket.put(source_file, target_file)


if __name__ == "__main__":
  upload_data()
