import hashlib
from unqlite import UnQLite
from utils.DimacsFile import DimacsFile

class BenchmarksFile:
    def __init__(self, filename="benchmarks.sqlite"):        
        db = UnQLite(filename=filename)
        table = db.collection("benchmarks")
        table.create()
        self.db = db
        self.table = table

    def _fetch_item_by_attribute(self, attr_name, attr_value):
        for item in self.table.all():
            if item.get(attr_name) == attr_value:
                return item
        return None

    def _deterministic_hash(self, input_string):
        # Create a new SHA-256 hash object
        sha256_hash = hashlib.sha256()

        # Update the hash object with the input string
        sha256_hash.update(input_string.encode("utf-8"))

        # Get the hexadecimal representation of the hash
        hash_result = sha256_hash.hexdigest()

        return hash_result


    def benchmarkFor(self, clauses):
        df = DimacsFile(clauses=clauses)
        clauses = df.clauses()  # converting tensor to list

        h = self._deterministic_hash(str(clauses))
        benchmark = self._fetch_item_by_attribute("hash", h)
        is_new = False
        if benchmark is None:            
            is_new = True
            benchmark = {}
            benchmark["hash"] = h
            # benchmark["clauses"] = clauses
        benchmark["n_vars"] = df.number_of_vars()
        benchmark["n_clauses"] = len(clauses)

        print("NEW" if is_new else "", "BENCHMARK: VARS=", df.number_of_vars(), " CLAUSES=", len(clauses))
        return benchmark
        
    def write(self, benchmark):
        self.db.begin()
        if "__id" in benchmark:
            print("UPDATING")
            self.table.update(benchmark["__id"], benchmark)
        else:
            self.table.store(benchmark)
        #        print(benchmark)
        print("BENCHMARK WRITTEN")
        self.db.commit()        

