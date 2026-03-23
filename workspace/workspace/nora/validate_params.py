#!/usr/bin/env python3
"""Comprehensive validation of all params.json files against docs/params_format.md spec."""

import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path("/Users/yifan/.thebotcompany/dev/src/github.com/sarchlab/gpu_benchmarks/repo")
VALID_TYPES = {"int", "float", "string", "enum"}
PARAM_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')

REQUIRED_TOP_FIELDS = {"benchmark", "tier", "suite", "description", "scaling_param", "non_scaling_params", "kernels"}
REQUIRED_SCALING_FIELDS = {"name", "description", "type", "values"}
REQUIRED_NON_SCALING_FIELDS = {"name", "description", "type", "default", "values"}


def find_params_files():
    files = sorted(REPO.glob("tier*/*/params.json"))
    return files


def validate_type_consistency(param_type, values, context):
    """Check that values match the declared type."""
    errors = []
    for i, v in enumerate(values):
        if param_type == "int":
            if not isinstance(v, int) or isinstance(v, bool):
                errors.append(f"{context}: value[{i}]={v!r} is not an int (type={type(v).__name__})")
        elif param_type == "float":
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errors.append(f"{context}: value[{i}]={v!r} is not a number (type={type(v).__name__})")
        elif param_type in ("string", "enum"):
            if not isinstance(v, str):
                errors.append(f"{context}: value[{i}]={v!r} is not a string (type={type(v).__name__})")
    return errors


def validate_file(filepath):
    errors = []
    warnings = []
    
    rel = filepath.relative_to(REPO)
    parts = rel.parts  # e.g., ('tier2', 'rodinia_hotspot', 'params.json')
    
    tier_dir = parts[0]   # 'tier2'
    bench_dir = parts[1]  # 'rodinia_hotspot'
    
    # Extract expected tier number
    try:
        expected_tier = int(tier_dir.replace("tier", ""))
    except ValueError:
        errors.append(f"Cannot parse tier from directory: {tier_dir}")
        expected_tier = None

    # Load JSON
    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return errors, warnings
    
    if not isinstance(data, dict):
        errors.append(f"Top-level is not a JSON object")
        return errors, warnings

    # Rule 9: Required top-level fields
    for field in REQUIRED_TOP_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
    
    # Rule 1: benchmark matches directory name
    if "benchmark" in data:
        if data["benchmark"] != bench_dir:
            errors.append(f"Rule 1: benchmark='{data['benchmark']}' does not match directory '{bench_dir}'")
    
    # Rule 2: tier matches parent directory
    if "tier" in data and expected_tier is not None:
        if data["tier"] != expected_tier:
            errors.append(f"Rule 2: tier={data['tier']} does not match directory '{tier_dir}' (expected {expected_tier})")
    
    # Validate scaling_param
    all_param_names = []
    
    if "scaling_param" in data:
        sp = data["scaling_param"]
        if not isinstance(sp, dict):
            errors.append(f"scaling_param is not an object")
        else:
            # Rule 12: scaling_param required subfields
            for field in REQUIRED_SCALING_FIELDS:
                if field not in sp:
                    errors.append(f"Rule 12: scaling_param missing required field: '{field}'")
            
            # Rule 7: parameter name is valid identifier
            if "name" in sp:
                all_param_names.append(sp["name"])
                if not PARAM_NAME_RE.match(sp["name"]):
                    errors.append(f"Rule 7: scaling_param name '{sp['name']}' is not a valid identifier ([a-z][a-z0-9_]*)")
            
            # Type field validation
            if "type" in sp:
                if sp["type"] not in VALID_TYPES:
                    errors.append(f"scaling_param type '{sp['type']}' not in {VALID_TYPES}")
            
            # Rule 3: scaling_param.values has 20-30 values (spec says "at least 10", recommended 20-30)
            if "values" in sp:
                vals = sp["values"]
                if not isinstance(vals, list):
                    errors.append(f"scaling_param.values is not an array")
                else:
                    if len(vals) < 10:
                        errors.append(f"Rule 3: scaling_param.values has {len(vals)} values (minimum 10)")
                    elif len(vals) < 20:
                        warnings.append(f"Rule 3: scaling_param.values has {len(vals)} values (recommended 20-30)")
                    elif len(vals) > 30:
                        warnings.append(f"Rule 3: scaling_param.values has {len(vals)} values (recommended 20-30)")
                    
                    # Rule 10: Type consistency
                    if "type" in sp:
                        errors.extend(validate_type_consistency(sp["type"], vals, "scaling_param"))
    
    # Validate non_scaling_params
    if "non_scaling_params" in data:
        nsp = data["non_scaling_params"]
        if not isinstance(nsp, list):
            errors.append(f"non_scaling_params is not an array")
        else:
            # Rule 11: At least 3 non_scaling_params
            if len(nsp) < 3:
                errors.append(f"Rule 11: non_scaling_params has {len(nsp)} entries (minimum 3 required)")
            
            for idx, param in enumerate(nsp):
                ctx = f"non_scaling_params[{idx}]"
                if not isinstance(param, dict):
                    errors.append(f"{ctx} is not an object")
                    continue
                
                param_name = param.get("name", f"<unnamed-{idx}>")
                ctx = f"non_scaling_params[{idx}] ('{param_name}')"
                
                # Rule 13: required subfields
                for field in REQUIRED_NON_SCALING_FIELDS:
                    if field not in param:
                        errors.append(f"Rule 13: {ctx} missing required field: '{field}'")
                
                # Rule 7: parameter name is valid identifier
                if "name" in param:
                    all_param_names.append(param["name"])
                    if not PARAM_NAME_RE.match(param["name"]):
                        errors.append(f"Rule 7: {ctx} name '{param['name']}' is not a valid identifier ([a-z][a-z0-9_]*)")
                
                # Type field validation
                if "type" in param:
                    if param["type"] not in VALID_TYPES:
                        errors.append(f"{ctx}: type '{param['type']}' not in {VALID_TYPES}")
                
                if "values" in param:
                    vals = param["values"]
                    if not isinstance(vals, list):
                        errors.append(f"{ctx}: values is not an array")
                    else:
                        # Rule 5: 3-5 values for non-scaling params
                        if len(vals) < 3 or len(vals) > 5:
                            errors.append(f"Rule 5: {ctx} has {len(vals)} values (expected 3-5)")
                        
                        # Rule 4: default appears in values
                        if "default" in param:
                            if param["default"] not in vals:
                                errors.append(f"Rule 4: {ctx} default={param['default']!r} not in values={vals!r}")
                        
                        # Rule 10: Type consistency for values
                        if "type" in param:
                            errors.extend(validate_type_consistency(param["type"], vals, ctx))
                
                # Rule 10: Type consistency for default
                if "default" in param and "type" in param:
                    d = param["default"]
                    pt = param["type"]
                    if pt == "int":
                        if not isinstance(d, int) or isinstance(d, bool):
                            errors.append(f"{ctx}: default={d!r} is not an int")
                    elif pt == "float":
                        if not isinstance(d, (int, float)) or isinstance(d, bool):
                            errors.append(f"{ctx}: default={d!r} is not a number")
                    elif pt in ("string", "enum"):
                        if not isinstance(d, str):
                            errors.append(f"{ctx}: default={d!r} is not a string")
    
    # Rule 8: No duplicate parameter names
    seen_names = set()
    for name in all_param_names:
        if name in seen_names:
            errors.append(f"Rule 8: Duplicate parameter name '{name}'")
        seen_names.add(name)
    
    # Rule 6: All kernels entries are unique strings
    if "kernels" in data:
        kernels = data["kernels"]
        if not isinstance(kernels, list):
            errors.append(f"kernels is not an array")
        else:
            seen_kernels = set()
            for i, k in enumerate(kernels):
                if not isinstance(k, str):
                    errors.append(f"Rule 6: kernels[{i}]={k!r} is not a string")
                elif k in seen_kernels:
                    errors.append(f"Rule 6: Duplicate kernel name '{k}'")
                seen_kernels.add(k)
    
    # Validate tier is an integer
    if "tier" in data:
        if not isinstance(data["tier"], int) or isinstance(data["tier"], bool):
            errors.append(f"tier={data['tier']!r} is not an integer")
    
    # Validate suite is a string
    if "suite" in data:
        if not isinstance(data["suite"], str):
            errors.append(f"suite is not a string")
    
    # Validate description is a string
    if "description" in data:
        if not isinstance(data["description"], str):
            errors.append(f"description is not a string")
    
    # Validate benchmark is a string
    if "benchmark" in data:
        if not isinstance(data["benchmark"], str):
            errors.append(f"benchmark is not a string")
    
    return errors, warnings


def main():
    files = find_params_files()
    print(f"Found {len(files)} params.json files\n")
    
    total_errors = 0
    total_warnings = 0
    files_with_errors = 0
    all_results = []
    
    error_summary = defaultdict(int)
    
    for f in files:
        rel = f.relative_to(REPO)
        errors, warnings = validate_file(f)
        
        if errors or warnings:
            all_results.append((rel, errors, warnings))
        
        if errors:
            files_with_errors += 1
            total_errors += len(errors)
            for e in errors:
                # Extract rule number if present
                m = re.match(r'Rule (\d+):', e)
                if m:
                    error_summary[f"Rule {m.group(1)}"] += 1
                else:
                    error_summary["Other"] += 1
        total_warnings += len(warnings)
    
    # Print results
    for rel, errors, warnings in all_results:
        print(f"=== {rel} ===")
        for e in errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  WARN:  {w}")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files:        {len(files)}")
    print(f"Files with errors:  {files_with_errors}")
    print(f"Files clean:        {len(files) - files_with_errors}")
    print(f"Total errors:       {total_errors}")
    print(f"Total warnings:     {total_warnings}")
    print()
    print("Errors by rule:")
    for rule, count in sorted(error_summary.items()):
        print(f"  {rule}: {count}")
    
    return 1 if total_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
