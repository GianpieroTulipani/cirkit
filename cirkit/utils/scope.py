class Scope(frozenset[int]):
    """An immutable container for a set of int to represent the scope of a node in the region
    graph or a layer in the circuit. A scope should always be a subset of range(num_variables),
    but for efficiency this is not checked.
    """

    def __iter__(self):
        """Ensure iteration happens in sorted order."""
        return iter(sorted(super().__iter__()))

    def __repr__(self) -> str:
        """Generate the repr string of the scope, for repr()."""
        return f"Scope({set(self)})"

    @classmethod
    def union(cls, *scopes: "Scope") -> "Scope":
        """Return a new Scope which is the union of all the given scopes."""
        all_vars: set[int] = set()
        for s in scopes:
            all_vars |= set(s)
        return cls(all_vars)
