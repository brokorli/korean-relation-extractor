from huggingface_hub import hf_hub_download


def download_file_from_hf_hub(repo_id: str, file_name: str):
    return hf_hub_download(repo_id=repo_id, filename=file_name)
