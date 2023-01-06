class NumberClustersOverflowError(Exception):
    """Custom error that is raised when the number of cluster is above the number of dwellings"""

    def __init__(self, nb_clusters: int, nb_dwellings: int, message: str) -> None:
        self.nb_clusters: int = nb_clusters
        self.nb_dwellings: int = nb_dwellings
        self.message: str = message
        super().__init__(message)
