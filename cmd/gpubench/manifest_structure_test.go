package main

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestValidateParamsRejectsMalformedStructure(t *testing.T) {
	cases := []struct {
		name, scope, field string
		value              interface{}
	}{
		{"suite number", "top", "suite", 1},
		{"description array", "top", "description", []interface{}{}},
		{"fractional tier", "top", "tier", 1.5},
		{"null scaling object", "top", "scaling_param", nil},
		{"array scaling object", "top", "scaling_param", []interface{}{}},
		{"string scaling object", "top", "scaling_param", "bad"},
		{"null non-scaling array", "top", "non_scaling_params", nil},
		{"object non-scaling array", "top", "non_scaling_params", map[string]interface{}{}},
		{"non-object parameter", "top", "non_scaling_params", []interface{}{1}},
		{"null parameter", "top", "non_scaling_params", []interface{}{nil}},
		{"non-string language", "top", "language", 1},
		{"unknown language", "top", "language", "rust"},
		{"non-array metrics", "top", "metrics", true},
		{"null metrics", "top", "metrics", nil},
		{"non-object metric", "top", "metrics", []interface{}{1}},
		{"missing metric description", "top", "metrics", []interface{}{map[string]interface{}{"name": "rate"}}},
		{"metric field type", "top", "metrics", []interface{}{map[string]interface{}{"name": 1, "description": "rate"}}},
		{"scaling name number", "scaling", "name", 1},
		{"scaling description number", "scaling", "description", 1},
		{"scaling values object", "scaling", "values", map[string]interface{}{}},
		{"scaling values null", "scaling", "values", nil},
		{"scaling values empty", "scaling", "values", []interface{}{}},
		{"scaling values descending", "scaling", "values", []interface{}{128, 64}},
		{"parameter name number", "parameter", "name", 1},
		{"parameter description number", "parameter", "description", 1},
		{"parameter values object", "parameter", "values", map[string]interface{}{}},
		{"parameter values null", "parameter", "values", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := baseValidParams("mybench")
			param := map[string]interface{}{
				"name": "block_size", "description": "threads", "type": "int",
				"default": 64, "values": []interface{}{64, 128},
			}
			p["non_scaling_params"] = []interface{}{param}
			target := p
			switch tc.scope {
			case "scaling":
				target = p["scaling_param"].(map[string]interface{})
			case "parameter":
				target = param
			}
			target[tc.field] = tc.value
			if errs, _ := validateParamsJSON(t, "mybench", p); len(errs) == 0 {
				t.Fatal("malformed manifest accepted")
			}
		})
	}
}

func TestValidateManifestFilesIncludesMalformedFiles(t *testing.T) {
	badObject := baseValidParams("bad")
	badObject["scaling_param"] = "invalid"
	badData, err := json.Marshal(badObject)
	if err != nil {
		t.Fatal(err)
	}
	goodData, err := json.Marshal(baseValidParams("good"))
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{"{", "null", "[]", "42", `"scalar"`, string(badData)} {
		t.Run(bad, func(t *testing.T) {
			root := t.TempDir()
			writeTestFile(t, filepath.Join(root, "tier1", "good", "params.json"), string(goodData))
			writeTestFile(t, filepath.Join(root, "tier1", "bad", "params.json"), bad)
			if got := validateManifestFiles(root, nil, "", ""); got != ExitSomeFailed {
				t.Fatalf("mixed valid/malformed manifests: exit=%d, want failure", got)
			}
			one := 1
			if got := validateManifestFiles(root, &one, "microbench", "bad"); got != ExitSomeFailed {
				t.Fatalf("selected malformed manifest: exit=%d, want failure", got)
			}
			if got := validateManifestFiles(root, &one, "microbench", "good"); got != ExitOK {
				t.Fatalf("selected valid manifest: exit=%d, want success", got)
			}
			if got := validateManifestFiles(root, nil, "", "absent"); got != ExitNoBench {
				t.Fatalf("unmatched filter: exit=%d, want no benchmarks", got)
			}
		})
	}
}

func TestValidateManifestFilesFiltersByPathAndSuite(t *testing.T) {
	root := t.TempDir()
	p := baseValidParams("claimed")
	p["suite"] = "other"
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	writeTestFile(t, filepath.Join(root, "tier1", "actual", "params.json"), string(data))
	if got := validateManifestFiles(root, nil, "", "actual"); got != ExitSomeFailed {
		t.Fatalf("benchmark field mismatch hidden by filter: exit=%d", got)
	}
	if got := validateManifestFiles(root, nil, "microbench", ""); got != ExitNoBench {
		t.Fatalf("other suite included: exit=%d", got)
	}
	two := 2
	if got := validateManifestFiles(root, &two, "", ""); got != ExitNoBench {
		t.Fatalf("other tier included: exit=%d", got)
	}
}

func TestValidateParamsKeepsShortAndCategoricalSweeps(t *testing.T) {
	for _, values := range [][]interface{}{{"large", "small"}, {"only"}} {
		p := baseValidParams("mybench")
		sp := p["scaling_param"].(map[string]interface{})
		sp["type"], sp["values"] = "enum", values
		if errs, warns := validateParamsJSON(t, "mybench", p); len(errs) != 0 || len(warns) == 0 {
			t.Fatalf("short categorical sweep: errors=%v warnings=%v", errs, warns)
		}
	}
}
