#!/usr/bin/env python3
"""
Create OpenSearch Serverless collection for movie vector search
Uses AWS Bedrock for embeddings
"""

import boto3
import json
import time
from datetime import datetime

def create_opensearch_serverless_collection():
    """Create OpenSearch Serverless collection with vector search support"""

    client = boto3.client('opensearchserverless', region_name='us-west-2')
    sts = boto3.client('sts', region_name='us-west-2')

    # Get current account info
    account_id = sts.get_caller_identity()['Account']
    collection_name = 'movie-search'

    print(f"Creating OpenSearch Serverless collection: {collection_name}")
    print(f"Account ID: {account_id}")

    # Step 1: Create encryption policy
    encryption_policy_name = f"{collection_name}-encrypt"
    encryption_policy = {
        "Rules": [
            {
                "ResourceType": "collection",
                "Resource": [f"collection/{collection_name}"]
            }
        ],
        "AWSOwnedKey": True
    }

    try:
        client.create_security_policy(
            name=encryption_policy_name,
            type='encryption',
            policy=json.dumps(encryption_policy)
        )
        print(f"✓ Created encryption policy: {encryption_policy_name}")
    except client.exceptions.ConflictException:
        print(f"✓ Encryption policy already exists: {encryption_policy_name}")
    except Exception as e:
        print(f"Error creating encryption policy: {e}")

    # Step 2: Create network policy (public access for demo)
    network_policy_name = f"{collection_name}-network"
    network_policy = [
        {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{collection_name}"]
                },
                {
                    "ResourceType": "dashboard",
                    "Resource": [f"collection/{collection_name}"]
                }
            ],
            "AllowFromPublic": True
        }
    ]

    try:
        client.create_security_policy(
            name=network_policy_name,
            type='network',
            policy=json.dumps(network_policy)
        )
        print(f"✓ Created network policy: {network_policy_name}")
    except client.exceptions.ConflictException:
        print(f"✓ Network policy already exists: {network_policy_name}")
    except Exception as e:
        print(f"Error creating network policy: {e}")

    # Step 3: Create data access policy
    data_policy_name = f"{collection_name}-data"

    # Get current IAM role ARN
    caller_identity = sts.get_caller_identity()
    principal_arn = caller_identity['Arn']

    data_policy = [
        {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{collection_name}"],
                    "Permission": [
                        "aoss:CreateCollectionItems",
                        "aoss:DeleteCollectionItems",
                        "aoss:UpdateCollectionItems",
                        "aoss:DescribeCollectionItems"
                    ]
                },
                {
                    "ResourceType": "index",
                    "Resource": [f"index/{collection_name}/*"],
                    "Permission": [
                        "aoss:CreateIndex",
                        "aoss:DeleteIndex",
                        "aoss:UpdateIndex",
                        "aoss:DescribeIndex",
                        "aoss:ReadDocument",
                        "aoss:WriteDocument"
                    ]
                }
            ],
            "Principal": [principal_arn]
        }
    ]

    try:
        client.create_access_policy(
            name=data_policy_name,
            type='data',
            policy=json.dumps(data_policy)
        )
        print(f"✓ Created data access policy: {data_policy_name}")
        print(f"  Principal: {principal_arn}")
    except client.exceptions.ConflictException:
        print(f"✓ Data access policy already exists: {data_policy_name}")
    except Exception as e:
        print(f"Error creating data access policy: {e}")

    # Step 4: Create the collection
    time.sleep(2)  # Wait for policies to propagate

    try:
        response = client.create_collection(
            name=collection_name,
            type='VECTORSEARCH',
            description='Movie search with semantic vector search'
        )
        collection_id = response['createCollectionDetail']['id']
        print(f"✓ Created collection: {collection_name}")
        print(f"  Collection ID: {collection_id}")
        print(f"  Status: {response['createCollectionDetail']['status']}")

        # Wait for collection to be active
        print("\nWaiting for collection to become active...")
        while True:
            status_response = client.batch_get_collection(names=[collection_name])
            if status_response['collectionDetails']:
                status = status_response['collectionDetails'][0]['status']
                print(f"  Current status: {status}")

                if status == 'ACTIVE':
                    endpoint = status_response['collectionDetails'][0]['collectionEndpoint']
                    dashboard_endpoint = status_response['collectionDetails'][0].get('dashboardEndpoint', 'N/A')

                    print(f"\n✅ Collection is ACTIVE!")
                    print(f"  Collection Endpoint: {endpoint}")
                    print(f"  Dashboard Endpoint: {dashboard_endpoint}")

                    # Save config
                    config = {
                        "collection_name": collection_name,
                        "collection_id": collection_id,
                        "endpoint": endpoint,
                        "dashboard_endpoint": dashboard_endpoint,
                        "region": "us-west-2",
                        "created_at": datetime.now().isoformat()
                    }

                    with open('config.json', 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"\n✓ Configuration saved to config.json")

                    return config

                elif status == 'FAILED':
                    print(f"❌ Collection creation failed!")
                    return None

            time.sleep(10)

    except client.exceptions.ConflictException:
        print(f"✓ Collection already exists: {collection_name}")
        # Get existing collection details
        response = client.batch_get_collection(names=[collection_name])
        if response['collectionDetails']:
            details = response['collectionDetails'][0]
            config = {
                "collection_name": collection_name,
                "collection_id": details['id'],
                "endpoint": details['collectionEndpoint'],
                "dashboard_endpoint": details.get('dashboardEndpoint', 'N/A'),
                "region": "us-west-2",
                "status": details['status']
            }
            print(f"  Endpoint: {config['endpoint']}")

            with open('config.json', 'w') as f:
                json.dump(config, f, indent=2)

            return config
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return None

if __name__ == "__main__":
    print("=" * 70)
    print("OpenSearch Serverless Collection Setup for Movie Search")
    print("=" * 70)
    print()

    config = create_opensearch_serverless_collection()

    if config:
        print("\n" + "=" * 70)
        print("✅ Setup Complete!")
        print("=" * 70)
        print(f"\nCollection: {config['collection_name']}")
        print(f"Endpoint: {config['endpoint']}")
        print("\nNext steps:")
        print("1. Run the movie indexer to load data")
        print("2. Start the Streamlit app for semantic search")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
