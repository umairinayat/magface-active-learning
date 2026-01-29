import numpy as np
from typing import List, Dict, Any, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, PointIdsList
)

from ..config_loader import get_config


class FaceIndex:

    def __init__(
        self,
        embedding_dim: int = 512,
        config_path: Optional[str] = None
    ):
        config = get_config(config_path)
        
        self.embedding_dim = embedding_dim
        self.url = config.index.url
        self.api_key = config.index.api_key
        self.collection_name = config.index.collection_name
        
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self._ensure_collection()
        
        print(f"FaceIndex connected to Qdrant at {self.url}, collection={self.collection_name}")

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )
            print(f"Created collection: {self.collection_name}")

    def _normalize(self, embedding: np.ndarray) -> List[float]:
        embedding = embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    def add(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        point_id = str(uuid.uuid4())
        vector = self._normalize(embedding)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=metadata or {})]
        )
        
        return point_id

    def add_batch(
        self,
        embeddings: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        points = []
        point_ids = []
        for i, emb in enumerate(embeddings):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            payload = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            points.append(PointStruct(id=point_id, vector=self._normalize(emb), payload=payload))
        
        self.client.upsert(collection_name=self.collection_name, points=points)
        return point_ids

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        vector = self._normalize(query_embedding)
        
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=k,
            score_threshold=threshold,
            with_payload=True
        )
        
        return [
            {
                'id': str(r.id),
                'score': r.score,
                'rank': i + 1,
                'metadata': r.payload or {}
            }
            for i, r in enumerate(response.points)
        ]

    def delete(self, point_ids: List[str]):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids)
        )

    def delete_by_filter(self, key: str, value: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key=key, match=MatchValue(value=value))]
            )
        )

    @property
    def size(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
