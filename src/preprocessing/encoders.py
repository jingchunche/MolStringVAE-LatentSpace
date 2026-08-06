"""Representation encoders and molecular enumeration strategies."""

import copy

import numpy as np
from rdkit import Chem


def _unique(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class MolecularEncoder:
    def __init__(self, max_attempts):
        self.max_attempts = max_attempts

    def encode(self, smiles, random_state, num_variants):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        canonical = self.canonical(mol)
        variants = self._enumerate_or_fallback(
            canonical,
            num_variants,
            lambda: self.enumerate(mol, canonical, random_state, num_variants),
        )
        return canonical, variants


    @staticmethod
    def _enumerate_or_fallback(canonical, num_variants, enumerate_variants):
        """Return no variants when enumeration fails, preserving canonical output."""
        try:
            variants = enumerate_variants()
            return _unique(
                value for value in variants if value != canonical
            )[:num_variants]
        except Exception:
            return []

    def canonical(self, mol):
        raise NotImplementedError

    def enumerate(self, mol, canonical, random_state, num_variants):
        if num_variants <= 0:
            return []

        variants = []
        seen = {canonical}
        for _ in range(self.max_attempts):
            try:
                atom_order = np.arange(mol.GetNumAtoms())
                if atom_order.size:
                    random_state.shuffle(atom_order)
                    enumerated_mol = Chem.RenumberAtoms(mol, atom_order.tolist())
                else:
                    enumerated_mol = mol
                value = self.encode_enumerated(enumerated_mol)
            except Exception:
                continue
            if value not in seen:
                seen.add(value)
                variants.append(value)
                if len(variants) >= num_variants:
                    break
        return variants

    def encode_enumerated(self, mol):
        raise NotImplementedError


class SmilesEncoder(MolecularEncoder):
    def canonical(self, mol):
        return Chem.MolToSmiles(mol, canonical=True)

    def encode_enumerated(self, mol):
        return Chem.MolToSmiles(mol, canonical=False)


class SelfiesEncoder(MolecularEncoder):
    def __init__(self, max_attempts):
        super().__init__(max_attempts)
        import selfies

        self.selfies = selfies

    def canonical(self, mol):
        return self.selfies.encoder(Chem.MolToSmiles(mol, canonical=True))

    def encode_enumerated(self, mol):
        smiles = Chem.MolToSmiles(mol, canonical=False)
        return self.selfies.encoder(smiles)


class GroupSelfiesEncoder(MolecularEncoder):
    def __init__(self, max_attempts, grammar_file, strategy, max_deleted_groups):
        super().__init__(max_attempts)
        from group_selfies.group_grammar import GroupGrammar

        self.grammar = GroupGrammar.from_file(grammar_file)
        self.strategy = strategy
        self.max_deleted_groups = max_deleted_groups

    def encode(self, smiles, random_state, num_variants):
        input_mol = Chem.MolFromSmiles(smiles)
        if input_mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        canonical_smiles = Chem.MolToSmiles(input_mol, canonical=True)
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES: {canonical_smiles}")

        canonical = self._encode_canonical(mol)
        variants = self._enumerate_or_fallback(
            canonical,
            num_variants,
            lambda: self._enumerate_variants(
                mol, canonical_smiles, canonical, random_state, num_variants
            ),
        )
        return canonical, variants

    def _enumerate_variants(
        self, mol, canonical_smiles, canonical, random_state, num_variants
    ):
        if num_variants <= 0 or self.strategy == "none":
            return []
        if self.strategy == "masking":
            return self._masking_variants(mol)
        return self._traversal_variants(
            mol, canonical_smiles, canonical, random_state, num_variants
        )

    def canonical(self, mol):
        return self._encode_canonical(mol)

    def encode_enumerated(self, mol):
        return self._encode_with_strategy(mol)

    def _encode_canonical(self, mol):
        if self.strategy == "masking":
            extracted = self.grammar.extract_groups(Chem.Mol(mol))
            return self.grammar.encoder(Chem.Mol(mol), extracted)
        return self.grammar.full_encoder(Chem.Mol(mol))

    @staticmethod
    def _group_names(extracted, limit):
        names = []
        for item in extracted:
            name = item[0].name
            if name not in names:
                names.append(name)
            if len(names) >= limit:
                break
        return names

    def _masking_variants(self, mol):
        extracted = self.grammar.extract_groups(Chem.Mol(mol))
        names = self._group_names(extracted, self.max_deleted_groups)
        deletion_sets = [(name,) for name in names]

        # Preserve the legacy M behavior: when only one group type is initially
        # present, try deleting a second group type exposed by the first deletion.
        if len(names) == 1 and self.max_deleted_groups > 1:
            grammar_once = copy.deepcopy(self.grammar)
            grammar_once.delete_group(names[0])
            extracted_once = grammar_once.extract_groups(Chem.Mol(mol))
            for second_name in self._group_names(extracted_once, 1):
                deletion_sets.append((names[0], second_name))

        variants = []
        for deletion_set in deletion_sets:
            grammar = copy.deepcopy(self.grammar)
            for name in deletion_set:
                grammar.delete_group(name)
            extracted_variant = grammar.extract_groups(Chem.Mol(mol))
            variants.append(grammar.encoder(Chem.Mol(mol), extracted_variant))
        return variants

    def _encode_with_strategy(self, mol):
        if self.strategy == "traversal":
            return self.grammar.full_encoder(mol)
        extracted = self.grammar.extract_groups(mol)
        if not extracted:
            return self.grammar.full_encoder(mol)
        grammar = copy.deepcopy(self.grammar)
        grammar.delete_group(extracted[0][0].name)
        extracted_variant = grammar.extract_groups(mol)
        return grammar.encoder(mol, extracted_variant)

    def _traversal_variants(
        self, mol, canonical_smiles, canonical, random_state, num_variants
    ):
        variants = []
        seen = {canonical}
        atom_indices = np.arange(mol.GetNumAtoms())

        for _ in range(self.max_attempts):
            random_state.shuffle(atom_indices)
            for root in atom_indices:
                if int(root) == 0:
                    continue
                try:
                    rooted_smiles = Chem.MolToSmiles(
                        mol,
                        rootedAtAtom=int(root),
                        canonical=False,
                        kekuleSmiles=True,
                    )
                    rooted_mol = Chem.MolFromSmiles(rooted_smiles)
                    if rooted_mol is None:
                        continue
                    if Chem.MolToSmiles(rooted_mol, canonical=True) != canonical_smiles:
                        continue
                    Chem.Kekulize(rooted_mol, clearAromaticFlags=True)
                    value = self._encode_with_strategy(rooted_mol)
                except Exception:
                    continue
                if value not in seen:
                    seen.add(value)
                    variants.append(value)
                    if len(variants) >= num_variants:
                        return variants
        return variants


def build_encoder(config):
    representation = config["representation"]
    max_attempts = config["enumeration"]["max_attempts"]
    if representation == "smiles":
        return SmilesEncoder(max_attempts)
    if representation == "selfies":
        return SelfiesEncoder(max_attempts)
    group_config = config["group_selfies"]
    return GroupSelfiesEncoder(
        max_attempts=max_attempts,
        grammar_file=group_config["grammar_file"],
        strategy=group_config["strategy"],
        max_deleted_groups=group_config["masking"]["max_deleted_groups"],
    )
