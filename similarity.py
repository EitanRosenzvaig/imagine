import os
import numpy as np
import glob
from logger import setup_custom_logger
from data import Data
from image_encoder import ImageEncoder
import boto3
from botocore.exceptions import ClientError
import zlib

log = setup_custom_logger('similarity')
LOCAL_PATH = 'live_products/'
TOTAL_SIMILARITIES_PER_PRODUCT = 1500
MIN_SIMILARITIES_EXPECTED = 100

STORAGE_SESSION = boto3.session.Session()
STORAGE_CLIENT = STORAGE_SESSION.client('s3',
                    region_name=os.environ.get('AWS_REGION_NAME'),
                    endpoint_url=os.environ.get('AWS_S3_ENDPOINT_URL'),
                    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)
SIMILARITY_BUCKET = os.environ.get('AWS_SIMILARITY_BUCKET_NAME')

def exists(file):
    return os.path.isfile(LOCAL_PATH + os.path.basename(file))

def download_from_s3(file, path_to_local_file):
    STORAGE_CLIENT.download_file(SIMILARITY_BUCKET, file, path_to_local_file)    

def download(file):
    try:
        file += '.gzip'
        path_to_local_file = LOCAL_PATH + os.path.basename(file)
        download_from_s3(file, path_to_local_file)
        compressed_image = open(path_to_local_file, 'rb').read()
        image = zlib.decompress(compressed_image)
        final_path_to_local_file = path_to_local_file.replace('.gzip','')
        f = open(final_path_to_local_file, 'wb')
        f.write(image)
        f.close()
        os.remove(path_to_local_file)
        return 1
    except ClientError as exc:
        if exc.response['Error']['Code'] == '404':
            log.error('File %s does not exist in similarity bucket', file)
        else:
            log.error('Error downloading %s', file, exc_info=True)
        return 0

def delete_old_products(live_products):
    live_files = [os.path.basename(product['fname']) for product in live_products]
    local_files = [os.path.basename(file) for file in glob.glob(LOCAL_PATH + '*')]
    local_files_to_remove = set(local_files) - set(live_files)
    for file in local_files_to_remove:
        os.remove(LOCAL_PATH + file)

def sync_local_files(live_products):
    delete_old_products(live_products)
    downloads = 0
    download_attempts = 0
    for product in live_products:
        file = product['fname']
        if not exists(file):
            downloads += download(file)
            download_attempts +=1
    log.info("Downloaded a total of %s files from a total of %s attempts. Failed on %s.", 
        downloads, 
        download_attempts,
        download_attempts - downloads
        )

def cosine_similarity(distances):
    sim = distances.dot(distances.T)
    if not isinstance(sim, np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def get_product_mapping(live_products):
    file_to_product_id = {}
    for product in live_products:
        fname = os.path.basename(product['fname'])
        file_to_product_id[fname] = product['id']
    return file_to_product_id

def get_similarities(distances, file_to_product_id, distance_idx_to_file):
    similarities = {}
    for i in range(distances.shape[0]):
        # Most similar first
        orderd_positions = distances[i,].argsort()[::-1][0:TOTAL_SIMILARITIES_PER_PRODUCT]
        if distance_idx_to_file[i] in file_to_product_id:
            similarities[file_to_product_id[distance_idx_to_file[i]]] = \
                            [
                                file_to_product_id[distance_idx_to_file[idx]] \
                                for idx in orderd_positions \
                                    if distance_idx_to_file[idx] in file_to_product_id \
                            ]
    return similarities

if __name__ == "__main__":
    data = Data()
    log.info("Getting live products")
    live_products = data.get_live_products()
    log.info("Found a total of %s live products", len(live_products))
    log.info("Starting local file sync with live products")
    sync_local_files(live_products)
    files = glob.glob(LOCAL_PATH + '*')
    image_encoder = ImageEncoder(files)
    encodings, distance_idx_to_file = image_encoder.get_encodings()
    log.info("Calculating similarities")
    distances = cosine_similarity(encodings)
    file_to_product_id = get_product_mapping(live_products)
    similarities = get_similarities(distances, file_to_product_id, distance_idx_to_file)
    if len(similarities) > MIN_SIMILARITIES_EXPECTED:
        log.info("Clearing similarities table")
        data.clear_similarity_table()
        log.info("Saving similarities")
        data.save_similarities(similarities.items())
    else:
        log.warning("Found only %s similarities, therefore, we will not refresh tables.", 
            len(similarities))
    data.close_connections()