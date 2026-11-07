from __future__ import annotations

# Fixed-endpoint hashes copied from the PR49 and PR52 published manifests.
# These are clean reconstructed TL3 roles, never original/all-labels weights.
PUBLISHED_RUNS = {
    "tl3c_fbc1k_s5": {
        "epoch": 3,
        "checkpoint_sha256": "840966894144bca5aafbf0a8e21ad0daab07a8503ba80e30d96668ef1bc3a496",
        "provenance_sha256": "b157c12556ec0f8cb60fc108a4e6ea28474ed9083d57032fa345a88e3818448c",
    },
    "tl3c_ffull_s3": {
        "epoch": 1,
        "checkpoint_sha256": "e52c2c6c9fd11f16ecc23b98710c30a73e15c08bc8d0b24c188fb07d10f439be",
        "provenance_sha256": "eaf99efbcc0f5fb09066ef9c15ca5966bc20327d15439bd1f1b8c0e3b208e3b3",
    },
    "tl3c_ffull_s4": {
        "epoch": 1,
        "checkpoint_sha256": "2873ed35e24aad6131b18375e45888556f69ace376c13251ea61b94dacacdeb2",
        "provenance_sha256": "997e16f532e98eac9ae320153c14d9f28002f5f0f64e8fb1afe65dc9d230a9a9",
    },
    "tl3c_fin256_s3": {
        "epoch": 3,
        "checkpoint_sha256": "36a99a47cde014f053f4cb8163ec7f3e1580a149c6d32a5db9c4d8045e9efef1",
        "provenance_sha256": "9a24ccc4f240d068bfed650776aa4c72d3356a79b5838120f737dabbe6d04f24",
    },
    "tl3c_fin256_s4": {
        "epoch": 3,
        "checkpoint_sha256": "3ce40c6f6a00e09cecfba0ac35566aa76c5791ddf761cb8a67cafbf1a38ae3d6",
        "provenance_sha256": "85cf341aa099267735c91766bd6633adfc45d089e72141bf752fab83371578d9",
    },
    "tl3c_fin256ft_s3": {
        "epoch": 1,
        "checkpoint_sha256": "3dd83448baf8b657a8d1782400e7d75f455e8b0d562c27b4d694d3c74103c090",
        "provenance_sha256": "5654ce519cdfa54913d6424bbd62a5f7ed40e59d87cd29a72f46a09e301eae09",
    },
    "tl3c_fin256ft_s4": {
        "epoch": 1,
        "checkpoint_sha256": "d1bf0f05082dcda72619740b63dbde5f0bedd4dbbbfe8963cba73e1e33bd3dda",
        "provenance_sha256": "a4c1915b5eab430d2f41ecd93ed21ebf89fd2c686e44abb2f020337232924a31",
    },
    "tl3c_fmu128_s3": {
        "epoch": 3,
        "checkpoint_sha256": "93ed16d83b93e989de42df33021a712e29455865531d794aef8936a0052f5390",
        "provenance_sha256": "4d1ae011643cd1cbbe6f2b72b402b55295e55821f2fd8a719cffba7f42889e24",
    },
    "tl3c_fmu128_s4": {
        "epoch": 3,
        "checkpoint_sha256": "abd6e016630d1b127cff527919d7bd53ac98313c814fb1f940b73ae301c7c427",
        "provenance_sha256": "6f3f2692a39490e810ace4c9d65aa38377847f14514487d782251b2918717c25",
    },
    "tl3c_frawW_s3": {
        "epoch": 3,
        "checkpoint_sha256": "e66a8cfa64f4667da540247bcec5575521f6fc6c261b378bd58d680b014cd088",
        "provenance_sha256": "8afcad3bb6125d92a3a906798731a1ad52fecf39197217c151396149ade9d06b",
    },
    "tl3c_frawW_s3ft": {
        "epoch": 1,
        "checkpoint_sha256": "190ca050a0caf6bc3b4a989520d62a64170453da124ad7e2b4e02ec3d7b15dcf",
        "provenance_sha256": "aff768ca4e5fd8565ff95ac88253bf757931701d89e09e7fec138678d954a095",
    },
    "tl3c_frawW_s4": {
        "epoch": 3,
        "checkpoint_sha256": "5b438245142a650e1a5654dca63fb508bad28b0c24ce5ca5570cbe1e8967589e",
        "provenance_sha256": "e9b21964b1abe8b87144c187d91729143ef62de82d0eaf6796ac5eab24a51279",
    },
    "tl3c_frawW_s4ft": {
        "epoch": 1,
        "checkpoint_sha256": "b69b3e0d822f8f0c20bcb5c20ad701d4a35e5a5e245bb1a6f5f3fe5239133d75",
        "provenance_sha256": "e1af4c554e804dec969f7fcb7d964213e3f70235c90ebdd8dfbab70ba0480905",
    },
    "tl3c_fttall_tlead_s5": {
        "epoch": 3,
        "checkpoint_sha256": "cf944dffac66d674894e4c589d839c407b3d28cb01d35a26f58ff08c7743f65f",
        "provenance_sha256": "fae7831bb2549162fe0c5d90e0f97f94619e7f9fd391e8b9efb71ecb821f6340",
    },
    "tl3c_fttall_tlead_s6": {
        "epoch": 3,
        "checkpoint_sha256": "24a565b03e35dc36679bf809f63f605f24f693404bd0cfda58bfc82177a33598",
        "provenance_sha256": "ad95ccc860b89c2bdb8f508a3fe74a05176a9e3226e8b9126eb65bc518161a52",
    },
    "tl3c_orgbase": {
        "epoch": 3,
        "checkpoint_sha256": "f188de19f3d5aed0cf9bf329d9e8aac218fc67c8e9afd6ca9f49a9cd4b3a8dec",
        "provenance_sha256": "905f1ebbc6acf5ca393cd1cf02b077260504bec99ddf687d7f0fcb5205ae6b2d",
    },
    "tl3c_w18final_distilled_s5": {
        "epoch": 1,
        "checkpoint_sha256": "a1934af24bd1520b78be874c294532b1e2d62acdd8bd1dc3d6b5c220b2dad107",
        "provenance_sha256": "3a67347d5c1133860c5cd3f910e296a0a5ee956aca54ec68290a763cf2946457",
    },
}

PR49_SOURCE_COMMIT = "c42baf7b95946994216801b4eb9909422ea7effa"
PR49_PREDICTOR_SHA256 = "981d7557082f8e33e025741dad2b2ab2e86fb20eb4347ab58eca1d53f00c9520"
PRIMARY_SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
COMPOSITION = [
    {"run": "tl3c_w18final_distilled_s5", "weight": 0.34, "epoch": 1},
    {"run": "tl3c_fttall_tlead_s5", "weight": 0.33, "epoch": 3},
    {"run": "tl3c_fttall_tlead_s6", "weight": 0.33, "epoch": 3},
]
ARITHMETIC = "fp32-left-to-right-0.34-0.33-0.33-no-clip"
ANCESTRY_LIMITATION = "published-legacy-consumed-cache-byte-ancestry-not-reconstructed"
