from cmflib.cmf import Cmf
#from ml_metadata.proto import metadata_store_pb2 as mlpb

print("Hello world")
metawriter=cmf.Cmf(
   filename="mlmd",                # Path to ML Metadata file.
   pipeline_name="smart_manufacturing",           # Name of a ML pipeline.
   graph=False)
_ = metawriter.create_context(pipeline_stage="Prepare", custom_properties={"user-metadata":"metadata_value"})