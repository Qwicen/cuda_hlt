// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_MODEL_NCATBOOSTFBS_H_
#define FLATBUFFERS_GENERATED_MODEL_NCATBOOSTFBS_H_

#include "flatbuffers/flatbuffers.h"

#include "ctr_data_generated.h"
#include "features_generated.h"

namespace NCatBoostFbs {

struct TKeyValue;

struct TObliviousTrees;

struct TModelCore;

inline const flatbuffers::TypeTable *TKeyValueTypeTable();

inline const flatbuffers::TypeTable *TObliviousTreesTypeTable();

inline const flatbuffers::TypeTable *TModelCoreTypeTable();

struct TKeyValue FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return TKeyValueTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_KEY = 4,
    VT_VALUE = 6
  };
  const flatbuffers::String *Key() const {
    return GetPointer<const flatbuffers::String *>(VT_KEY);
  }
  flatbuffers::String *mutable_Key() {
    return GetPointer<flatbuffers::String *>(VT_KEY);
  }
  bool KeyCompareLessThan(const TKeyValue *o) const {
    return *Key() < *o->Key();
  }
  int KeyCompareWithValue(const char *val) const {
    return strcmp(Key()->c_str(), val);
  }
  const flatbuffers::String *Value() const {
    return GetPointer<const flatbuffers::String *>(VT_VALUE);
  }
  flatbuffers::String *mutable_Value() {
    return GetPointer<flatbuffers::String *>(VT_VALUE);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffsetRequired(verifier, VT_KEY) &&
           verifier.VerifyString(Key()) &&
           VerifyOffsetRequired(verifier, VT_VALUE) &&
           verifier.VerifyString(Value()) &&
           verifier.EndTable();
  }
};

struct TKeyValueBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_Key(flatbuffers::Offset<flatbuffers::String> Key) {
    fbb_.AddOffset(TKeyValue::VT_KEY, Key);
  }
  void add_Value(flatbuffers::Offset<flatbuffers::String> Value) {
    fbb_.AddOffset(TKeyValue::VT_VALUE, Value);
  }
  explicit TKeyValueBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  TKeyValueBuilder &operator=(const TKeyValueBuilder &);
  flatbuffers::Offset<TKeyValue> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<TKeyValue>(end);
    fbb_.Required(o, TKeyValue::VT_KEY);
    fbb_.Required(o, TKeyValue::VT_VALUE);
    return o;
  }
};

inline flatbuffers::Offset<TKeyValue> CreateTKeyValue(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> Key = 0,
    flatbuffers::Offset<flatbuffers::String> Value = 0) {
  TKeyValueBuilder builder_(_fbb);
  builder_.add_Value(Value);
  builder_.add_Key(Key);
  return builder_.Finish();
}

inline flatbuffers::Offset<TKeyValue> CreateTKeyValueDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *Key = nullptr,
    const char *Value = nullptr) {
  auto Key__ = Key ? _fbb.CreateString(Key) : 0;
  auto Value__ = Value ? _fbb.CreateString(Value) : 0;
  return NCatBoostFbs::CreateTKeyValue(
      _fbb,
      Key__,
      Value__);
}

struct TObliviousTrees FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return TObliviousTreesTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_APPROXDIMENSION = 4,
    VT_TREESPLITS = 6,
    VT_TREESIZES = 8,
    VT_TREESTARTOFFSETS = 10,
    VT_CATFEATURES = 12,
    VT_FLOATFEATURES = 14,
    VT_ONEHOTFEATURES = 16,
    VT_CTRFEATURES = 18,
    VT_LEAFVALUES = 20,
    VT_LEAFWEIGHTS = 22
  };
  int32_t ApproxDimension() const {
    return GetField<int32_t>(VT_APPROXDIMENSION, 0);
  }
  bool mutate_ApproxDimension(int32_t _ApproxDimension) {
    return SetField<int32_t>(VT_APPROXDIMENSION, _ApproxDimension, 0);
  }
  const flatbuffers::Vector<int32_t> *TreeSplits() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_TREESPLITS);
  }
  flatbuffers::Vector<int32_t> *mutable_TreeSplits() {
    return GetPointer<flatbuffers::Vector<int32_t> *>(VT_TREESPLITS);
  }
  const flatbuffers::Vector<int32_t> *TreeSizes() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_TREESIZES);
  }
  flatbuffers::Vector<int32_t> *mutable_TreeSizes() {
    return GetPointer<flatbuffers::Vector<int32_t> *>(VT_TREESIZES);
  }
  const flatbuffers::Vector<int32_t> *TreeStartOffsets() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(VT_TREESTARTOFFSETS);
  }
  flatbuffers::Vector<int32_t> *mutable_TreeStartOffsets() {
    return GetPointer<flatbuffers::Vector<int32_t> *>(VT_TREESTARTOFFSETS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<TCatFeature>> *CatFeatures() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<TCatFeature>> *>(VT_CATFEATURES);
  }
  flatbuffers::Vector<flatbuffers::Offset<TCatFeature>> *mutable_CatFeatures() {
    return GetPointer<flatbuffers::Vector<flatbuffers::Offset<TCatFeature>> *>(VT_CATFEATURES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<TFloatFeature>> *FloatFeatures() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<TFloatFeature>> *>(VT_FLOATFEATURES);
  }
  flatbuffers::Vector<flatbuffers::Offset<TFloatFeature>> *mutable_FloatFeatures() {
    return GetPointer<flatbuffers::Vector<flatbuffers::Offset<TFloatFeature>> *>(VT_FLOATFEATURES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<TOneHotFeature>> *OneHotFeatures() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<TOneHotFeature>> *>(VT_ONEHOTFEATURES);
  }
  flatbuffers::Vector<flatbuffers::Offset<TOneHotFeature>> *mutable_OneHotFeatures() {
    return GetPointer<flatbuffers::Vector<flatbuffers::Offset<TOneHotFeature>> *>(VT_ONEHOTFEATURES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<TCtrFeature>> *CtrFeatures() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<TCtrFeature>> *>(VT_CTRFEATURES);
  }
  flatbuffers::Vector<flatbuffers::Offset<TCtrFeature>> *mutable_CtrFeatures() {
    return GetPointer<flatbuffers::Vector<flatbuffers::Offset<TCtrFeature>> *>(VT_CTRFEATURES);
  }
  const flatbuffers::Vector<double> *LeafValues() const {
    return GetPointer<const flatbuffers::Vector<double> *>(VT_LEAFVALUES);
  }
  flatbuffers::Vector<double> *mutable_LeafValues() {
    return GetPointer<flatbuffers::Vector<double> *>(VT_LEAFVALUES);
  }
  const flatbuffers::Vector<double> *LeafWeights() const {
    return GetPointer<const flatbuffers::Vector<double> *>(VT_LEAFWEIGHTS);
  }
  flatbuffers::Vector<double> *mutable_LeafWeights() {
    return GetPointer<flatbuffers::Vector<double> *>(VT_LEAFWEIGHTS);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int32_t>(verifier, VT_APPROXDIMENSION) &&
           VerifyOffset(verifier, VT_TREESPLITS) &&
           verifier.VerifyVector(TreeSplits()) &&
           VerifyOffset(verifier, VT_TREESIZES) &&
           verifier.VerifyVector(TreeSizes()) &&
           VerifyOffset(verifier, VT_TREESTARTOFFSETS) &&
           verifier.VerifyVector(TreeStartOffsets()) &&
           VerifyOffset(verifier, VT_CATFEATURES) &&
           verifier.VerifyVector(CatFeatures()) &&
           verifier.VerifyVectorOfTables(CatFeatures()) &&
           VerifyOffset(verifier, VT_FLOATFEATURES) &&
           verifier.VerifyVector(FloatFeatures()) &&
           verifier.VerifyVectorOfTables(FloatFeatures()) &&
           VerifyOffset(verifier, VT_ONEHOTFEATURES) &&
           verifier.VerifyVector(OneHotFeatures()) &&
           verifier.VerifyVectorOfTables(OneHotFeatures()) &&
           VerifyOffset(verifier, VT_CTRFEATURES) &&
           verifier.VerifyVector(CtrFeatures()) &&
           verifier.VerifyVectorOfTables(CtrFeatures()) &&
           VerifyOffset(verifier, VT_LEAFVALUES) &&
           verifier.VerifyVector(LeafValues()) &&
           VerifyOffset(verifier, VT_LEAFWEIGHTS) &&
           verifier.VerifyVector(LeafWeights()) &&
           verifier.EndTable();
  }
};

struct TObliviousTreesBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_ApproxDimension(int32_t ApproxDimension) {
    fbb_.AddElement<int32_t>(TObliviousTrees::VT_APPROXDIMENSION, ApproxDimension, 0);
  }
  void add_TreeSplits(flatbuffers::Offset<flatbuffers::Vector<int32_t>> TreeSplits) {
    fbb_.AddOffset(TObliviousTrees::VT_TREESPLITS, TreeSplits);
  }
  void add_TreeSizes(flatbuffers::Offset<flatbuffers::Vector<int32_t>> TreeSizes) {
    fbb_.AddOffset(TObliviousTrees::VT_TREESIZES, TreeSizes);
  }
  void add_TreeStartOffsets(flatbuffers::Offset<flatbuffers::Vector<int32_t>> TreeStartOffsets) {
    fbb_.AddOffset(TObliviousTrees::VT_TREESTARTOFFSETS, TreeStartOffsets);
  }
  void add_CatFeatures(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TCatFeature>>> CatFeatures) {
    fbb_.AddOffset(TObliviousTrees::VT_CATFEATURES, CatFeatures);
  }
  void add_FloatFeatures(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TFloatFeature>>> FloatFeatures) {
    fbb_.AddOffset(TObliviousTrees::VT_FLOATFEATURES, FloatFeatures);
  }
  void add_OneHotFeatures(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TOneHotFeature>>> OneHotFeatures) {
    fbb_.AddOffset(TObliviousTrees::VT_ONEHOTFEATURES, OneHotFeatures);
  }
  void add_CtrFeatures(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TCtrFeature>>> CtrFeatures) {
    fbb_.AddOffset(TObliviousTrees::VT_CTRFEATURES, CtrFeatures);
  }
  void add_LeafValues(flatbuffers::Offset<flatbuffers::Vector<double>> LeafValues) {
    fbb_.AddOffset(TObliviousTrees::VT_LEAFVALUES, LeafValues);
  }
  void add_LeafWeights(flatbuffers::Offset<flatbuffers::Vector<double>> LeafWeights) {
    fbb_.AddOffset(TObliviousTrees::VT_LEAFWEIGHTS, LeafWeights);
  }
  explicit TObliviousTreesBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  TObliviousTreesBuilder &operator=(const TObliviousTreesBuilder &);
  flatbuffers::Offset<TObliviousTrees> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<TObliviousTrees>(end);
    return o;
  }
};

inline flatbuffers::Offset<TObliviousTrees> CreateTObliviousTrees(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t ApproxDimension = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> TreeSplits = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> TreeSizes = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> TreeStartOffsets = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TCatFeature>>> CatFeatures = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TFloatFeature>>> FloatFeatures = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TOneHotFeature>>> OneHotFeatures = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TCtrFeature>>> CtrFeatures = 0,
    flatbuffers::Offset<flatbuffers::Vector<double>> LeafValues = 0,
    flatbuffers::Offset<flatbuffers::Vector<double>> LeafWeights = 0) {
  TObliviousTreesBuilder builder_(_fbb);
  builder_.add_LeafWeights(LeafWeights);
  builder_.add_LeafValues(LeafValues);
  builder_.add_CtrFeatures(CtrFeatures);
  builder_.add_OneHotFeatures(OneHotFeatures);
  builder_.add_FloatFeatures(FloatFeatures);
  builder_.add_CatFeatures(CatFeatures);
  builder_.add_TreeStartOffsets(TreeStartOffsets);
  builder_.add_TreeSizes(TreeSizes);
  builder_.add_TreeSplits(TreeSplits);
  builder_.add_ApproxDimension(ApproxDimension);
  return builder_.Finish();
}

inline flatbuffers::Offset<TObliviousTrees> CreateTObliviousTreesDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    int32_t ApproxDimension = 0,
    const std::vector<int32_t> *TreeSplits = nullptr,
    const std::vector<int32_t> *TreeSizes = nullptr,
    const std::vector<int32_t> *TreeStartOffsets = nullptr,
    const std::vector<flatbuffers::Offset<TCatFeature>> *CatFeatures = nullptr,
    const std::vector<flatbuffers::Offset<TFloatFeature>> *FloatFeatures = nullptr,
    const std::vector<flatbuffers::Offset<TOneHotFeature>> *OneHotFeatures = nullptr,
    const std::vector<flatbuffers::Offset<TCtrFeature>> *CtrFeatures = nullptr,
    const std::vector<double> *LeafValues = nullptr,
    const std::vector<double> *LeafWeights = nullptr) {
  auto TreeSplits__ = TreeSplits ? _fbb.CreateVector<int32_t>(*TreeSplits) : 0;
  auto TreeSizes__ = TreeSizes ? _fbb.CreateVector<int32_t>(*TreeSizes) : 0;
  auto TreeStartOffsets__ = TreeStartOffsets ? _fbb.CreateVector<int32_t>(*TreeStartOffsets) : 0;
  auto CatFeatures__ = CatFeatures ? _fbb.CreateVector<flatbuffers::Offset<TCatFeature>>(*CatFeatures) : 0;
  auto FloatFeatures__ = FloatFeatures ? _fbb.CreateVector<flatbuffers::Offset<TFloatFeature>>(*FloatFeatures) : 0;
  auto OneHotFeatures__ = OneHotFeatures ? _fbb.CreateVector<flatbuffers::Offset<TOneHotFeature>>(*OneHotFeatures) : 0;
  auto CtrFeatures__ = CtrFeatures ? _fbb.CreateVector<flatbuffers::Offset<TCtrFeature>>(*CtrFeatures) : 0;
  auto LeafValues__ = LeafValues ? _fbb.CreateVector<double>(*LeafValues) : 0;
  auto LeafWeights__ = LeafWeights ? _fbb.CreateVector<double>(*LeafWeights) : 0;
  return NCatBoostFbs::CreateTObliviousTrees(
      _fbb,
      ApproxDimension,
      TreeSplits__,
      TreeSizes__,
      TreeStartOffsets__,
      CatFeatures__,
      FloatFeatures__,
      OneHotFeatures__,
      CtrFeatures__,
      LeafValues__,
      LeafWeights__);
}

struct TModelCore FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return TModelCoreTypeTable();
  }
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_FORMATVERSION = 4,
    VT_OBLIVIOUSTREES = 6,
    VT_INFOMAP = 8,
    VT_MODELPARTIDS = 10
  };
  const flatbuffers::String *FormatVersion() const {
    return GetPointer<const flatbuffers::String *>(VT_FORMATVERSION);
  }
  flatbuffers::String *mutable_FormatVersion() {
    return GetPointer<flatbuffers::String *>(VT_FORMATVERSION);
  }
  const TObliviousTrees *ObliviousTrees() const {
    return GetPointer<const TObliviousTrees *>(VT_OBLIVIOUSTREES);
  }
  TObliviousTrees *mutable_ObliviousTrees() {
    return GetPointer<TObliviousTrees *>(VT_OBLIVIOUSTREES);
  }
  const flatbuffers::Vector<flatbuffers::Offset<TKeyValue>> *InfoMap() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<TKeyValue>> *>(VT_INFOMAP);
  }
  flatbuffers::Vector<flatbuffers::Offset<TKeyValue>> *mutable_InfoMap() {
    return GetPointer<flatbuffers::Vector<flatbuffers::Offset<TKeyValue>> *>(VT_INFOMAP);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *ModelPartIds() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_MODELPARTIDS);
  }
  flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *mutable_ModelPartIds() {
    return GetPointer<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_MODELPARTIDS);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_FORMATVERSION) &&
           verifier.VerifyString(FormatVersion()) &&
           VerifyOffset(verifier, VT_OBLIVIOUSTREES) &&
           verifier.VerifyTable(ObliviousTrees()) &&
           VerifyOffset(verifier, VT_INFOMAP) &&
           verifier.VerifyVector(InfoMap()) &&
           verifier.VerifyVectorOfTables(InfoMap()) &&
           VerifyOffset(verifier, VT_MODELPARTIDS) &&
           verifier.VerifyVector(ModelPartIds()) &&
           verifier.VerifyVectorOfStrings(ModelPartIds()) &&
           verifier.EndTable();
  }
};

struct TModelCoreBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_FormatVersion(flatbuffers::Offset<flatbuffers::String> FormatVersion) {
    fbb_.AddOffset(TModelCore::VT_FORMATVERSION, FormatVersion);
  }
  void add_ObliviousTrees(flatbuffers::Offset<TObliviousTrees> ObliviousTrees) {
    fbb_.AddOffset(TModelCore::VT_OBLIVIOUSTREES, ObliviousTrees);
  }
  void add_InfoMap(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TKeyValue>>> InfoMap) {
    fbb_.AddOffset(TModelCore::VT_INFOMAP, InfoMap);
  }
  void add_ModelPartIds(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> ModelPartIds) {
    fbb_.AddOffset(TModelCore::VT_MODELPARTIDS, ModelPartIds);
  }
  explicit TModelCoreBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  TModelCoreBuilder &operator=(const TModelCoreBuilder &);
  flatbuffers::Offset<TModelCore> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<TModelCore>(end);
    return o;
  }
};

inline flatbuffers::Offset<TModelCore> CreateTModelCore(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::String> FormatVersion = 0,
    flatbuffers::Offset<TObliviousTrees> ObliviousTrees = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<TKeyValue>>> InfoMap = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> ModelPartIds = 0) {
  TModelCoreBuilder builder_(_fbb);
  builder_.add_ModelPartIds(ModelPartIds);
  builder_.add_InfoMap(InfoMap);
  builder_.add_ObliviousTrees(ObliviousTrees);
  builder_.add_FormatVersion(FormatVersion);
  return builder_.Finish();
}

inline flatbuffers::Offset<TModelCore> CreateTModelCoreDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const char *FormatVersion = nullptr,
    flatbuffers::Offset<TObliviousTrees> ObliviousTrees = 0,
    const std::vector<flatbuffers::Offset<TKeyValue>> *InfoMap = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *ModelPartIds = nullptr) {
  auto FormatVersion__ = FormatVersion ? _fbb.CreateString(FormatVersion) : 0;
  auto InfoMap__ = InfoMap ? _fbb.CreateVector<flatbuffers::Offset<TKeyValue>>(*InfoMap) : 0;
  auto ModelPartIds__ = ModelPartIds ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*ModelPartIds) : 0;
  return NCatBoostFbs::CreateTModelCore(
      _fbb,
      FormatVersion__,
      ObliviousTrees,
      InfoMap__,
      ModelPartIds__);
}

inline const flatbuffers::TypeTable *TKeyValueTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_STRING, 0, -1 },
    { flatbuffers::ET_STRING, 0, -1 }
  };
  static const char * const names[] = {
    "Key",
    "Value"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 2, type_codes, nullptr, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *TObliviousTreesTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_INT, 0, -1 },
    { flatbuffers::ET_INT, 1, -1 },
    { flatbuffers::ET_INT, 1, -1 },
    { flatbuffers::ET_INT, 1, -1 },
    { flatbuffers::ET_SEQUENCE, 1, 0 },
    { flatbuffers::ET_SEQUENCE, 1, 1 },
    { flatbuffers::ET_SEQUENCE, 1, 2 },
    { flatbuffers::ET_SEQUENCE, 1, 3 },
    { flatbuffers::ET_DOUBLE, 1, -1 },
    { flatbuffers::ET_DOUBLE, 1, -1 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    TCatFeatureTypeTable,
    TFloatFeatureTypeTable,
    TOneHotFeatureTypeTable,
    TCtrFeatureTypeTable
  };
  static const char * const names[] = {
    "ApproxDimension",
    "TreeSplits",
    "TreeSizes",
    "TreeStartOffsets",
    "CatFeatures",
    "FloatFeatures",
    "OneHotFeatures",
    "CtrFeatures",
    "LeafValues",
    "LeafWeights"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 10, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *TModelCoreTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_STRING, 0, -1 },
    { flatbuffers::ET_SEQUENCE, 0, 0 },
    { flatbuffers::ET_SEQUENCE, 1, 1 },
    { flatbuffers::ET_STRING, 1, -1 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    TObliviousTreesTypeTable,
    TKeyValueTypeTable
  };
  static const char * const names[] = {
    "FormatVersion",
    "ObliviousTrees",
    "InfoMap",
    "ModelPartIds"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 4, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

inline const NCatBoostFbs::TModelCore *GetTModelCore(const void *buf) {
  return flatbuffers::GetRoot<NCatBoostFbs::TModelCore>(buf);
}

inline const NCatBoostFbs::TModelCore *GetSizePrefixedTModelCore(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<NCatBoostFbs::TModelCore>(buf);
}

inline TModelCore *GetMutableTModelCore(void *buf) {
  return flatbuffers::GetMutableRoot<TModelCore>(buf);
}

inline bool VerifyTModelCoreBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<NCatBoostFbs::TModelCore>(nullptr);
}

inline bool VerifySizePrefixedTModelCoreBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<NCatBoostFbs::TModelCore>(nullptr);
}

inline void FinishTModelCoreBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<NCatBoostFbs::TModelCore> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedTModelCoreBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<NCatBoostFbs::TModelCore> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace NCatBoostFbs

#endif  // FLATBUFFERS_GENERATED_MODEL_NCATBOOSTFBS_H_
