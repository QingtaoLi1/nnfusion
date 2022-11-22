// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: nnf_tensor_shape.proto

#ifndef PROTOBUF_INCLUDED_nnf_5ftensor_5fshape_2eproto
#define PROTOBUF_INCLUDED_nnf_5ftensor_5fshape_2eproto

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3006001
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3006001 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#define PROTOBUF_INTERNAL_EXPORT_protobuf_nnf_5ftensor_5fshape_2eproto 

namespace protobuf_nnf_5ftensor_5fshape_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[2];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
}  // namespace protobuf_nnf_5ftensor_5fshape_2eproto
namespace nnfusion {
namespace serialize {
class TensorShapeProto;
class TensorShapeProtoDefaultTypeInternal;
extern TensorShapeProtoDefaultTypeInternal _TensorShapeProto_default_instance_;
class TensorShapeProto_Dim;
class TensorShapeProto_DimDefaultTypeInternal;
extern TensorShapeProto_DimDefaultTypeInternal _TensorShapeProto_Dim_default_instance_;
}  // namespace serialize
}  // namespace nnfusion
namespace google {
namespace protobuf {
template<> ::nnfusion::serialize::TensorShapeProto* Arena::CreateMaybeMessage<::nnfusion::serialize::TensorShapeProto>(Arena*);
template<> ::nnfusion::serialize::TensorShapeProto_Dim* Arena::CreateMaybeMessage<::nnfusion::serialize::TensorShapeProto_Dim>(Arena*);
}  // namespace protobuf
}  // namespace google
namespace nnfusion {
namespace serialize {

// ===================================================================

class TensorShapeProto_Dim : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:nnfusion.serialize.TensorShapeProto.Dim) */ {
 public:
  TensorShapeProto_Dim();
  virtual ~TensorShapeProto_Dim();

  TensorShapeProto_Dim(const TensorShapeProto_Dim& from);

  inline TensorShapeProto_Dim& operator=(const TensorShapeProto_Dim& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  TensorShapeProto_Dim(TensorShapeProto_Dim&& from) noexcept
    : TensorShapeProto_Dim() {
    *this = ::std::move(from);
  }

  inline TensorShapeProto_Dim& operator=(TensorShapeProto_Dim&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline ::google::protobuf::Arena* GetArena() const final {
    return GetArenaNoVirtual();
  }
  inline void* GetMaybeArenaPointer() const final {
    return MaybeArenaPtr();
  }
  static const ::google::protobuf::Descriptor* descriptor();
  static const TensorShapeProto_Dim& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const TensorShapeProto_Dim* internal_default_instance() {
    return reinterpret_cast<const TensorShapeProto_Dim*>(
               &_TensorShapeProto_Dim_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void UnsafeArenaSwap(TensorShapeProto_Dim* other);
  void Swap(TensorShapeProto_Dim* other);
  friend void swap(TensorShapeProto_Dim& a, TensorShapeProto_Dim& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline TensorShapeProto_Dim* New() const final {
    return CreateMaybeMessage<TensorShapeProto_Dim>(NULL);
  }

  TensorShapeProto_Dim* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<TensorShapeProto_Dim>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const TensorShapeProto_Dim& from);
  void MergeFrom(const TensorShapeProto_Dim& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(TensorShapeProto_Dim* other);
  protected:
  explicit TensorShapeProto_Dim(::google::protobuf::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::google::protobuf::Arena* arena);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // string name = 2;
  void clear_name();
  static const int kNameFieldNumber = 2;
  const ::std::string& name() const;
  void set_name(const ::std::string& value);
  #if LANG_CXX11
  void set_name(::std::string&& value);
  #endif
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  ::std::string* mutable_name();
  ::std::string* release_name();
  void set_allocated_name(::std::string* name);
  PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  ::std::string* unsafe_arena_release_name();
  PROTOBUF_RUNTIME_DEPRECATED("The unsafe_arena_ accessors for"
  "    string fields are deprecated and will be removed in a"
  "    future release.")
  void unsafe_arena_set_allocated_name(
      ::std::string* name);

  // int64 size = 1;
  void clear_size();
  static const int kSizeFieldNumber = 1;
  ::google::protobuf::int64 size() const;
  void set_size(::google::protobuf::int64 value);

  // @@protoc_insertion_point(class_scope:nnfusion.serialize.TensorShapeProto.Dim)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::internal::ArenaStringPtr name_;
  ::google::protobuf::int64 size_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_nnf_5ftensor_5fshape_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class TensorShapeProto : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:nnfusion.serialize.TensorShapeProto) */ {
 public:
  TensorShapeProto();
  virtual ~TensorShapeProto();

  TensorShapeProto(const TensorShapeProto& from);

  inline TensorShapeProto& operator=(const TensorShapeProto& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  TensorShapeProto(TensorShapeProto&& from) noexcept
    : TensorShapeProto() {
    *this = ::std::move(from);
  }

  inline TensorShapeProto& operator=(TensorShapeProto&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline ::google::protobuf::Arena* GetArena() const final {
    return GetArenaNoVirtual();
  }
  inline void* GetMaybeArenaPointer() const final {
    return MaybeArenaPtr();
  }
  static const ::google::protobuf::Descriptor* descriptor();
  static const TensorShapeProto& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const TensorShapeProto* internal_default_instance() {
    return reinterpret_cast<const TensorShapeProto*>(
               &_TensorShapeProto_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void UnsafeArenaSwap(TensorShapeProto* other);
  void Swap(TensorShapeProto* other);
  friend void swap(TensorShapeProto& a, TensorShapeProto& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline TensorShapeProto* New() const final {
    return CreateMaybeMessage<TensorShapeProto>(NULL);
  }

  TensorShapeProto* New(::google::protobuf::Arena* arena) const final {
    return CreateMaybeMessage<TensorShapeProto>(arena);
  }
  void CopyFrom(const ::google::protobuf::Message& from) final;
  void MergeFrom(const ::google::protobuf::Message& from) final;
  void CopyFrom(const TensorShapeProto& from);
  void MergeFrom(const TensorShapeProto& from);
  void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) final;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const final;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(TensorShapeProto* other);
  protected:
  explicit TensorShapeProto(::google::protobuf::Arena* arena);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::google::protobuf::Arena* arena);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef TensorShapeProto_Dim Dim;

  // accessors -------------------------------------------------------

  // repeated .nnfusion.serialize.TensorShapeProto.Dim dim = 2;
  int dim_size() const;
  void clear_dim();
  static const int kDimFieldNumber = 2;
  ::nnfusion::serialize::TensorShapeProto_Dim* mutable_dim(int index);
  ::google::protobuf::RepeatedPtrField< ::nnfusion::serialize::TensorShapeProto_Dim >*
      mutable_dim();
  const ::nnfusion::serialize::TensorShapeProto_Dim& dim(int index) const;
  ::nnfusion::serialize::TensorShapeProto_Dim* add_dim();
  const ::google::protobuf::RepeatedPtrField< ::nnfusion::serialize::TensorShapeProto_Dim >&
      dim() const;

  // bool unknown_rank = 3;
  void clear_unknown_rank();
  static const int kUnknownRankFieldNumber = 3;
  bool unknown_rank() const;
  void set_unknown_rank(bool value);

  // @@protoc_insertion_point(class_scope:nnfusion.serialize.TensorShapeProto)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  template <typename T> friend class ::google::protobuf::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::google::protobuf::RepeatedPtrField< ::nnfusion::serialize::TensorShapeProto_Dim > dim_;
  bool unknown_rank_;
  mutable ::google::protobuf::internal::CachedSize _cached_size_;
  friend struct ::protobuf_nnf_5ftensor_5fshape_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// TensorShapeProto_Dim

// int64 size = 1;
inline void TensorShapeProto_Dim::clear_size() {
  size_ = GOOGLE_LONGLONG(0);
}
inline ::google::protobuf::int64 TensorShapeProto_Dim::size() const {
  // @@protoc_insertion_point(field_get:nnfusion.serialize.TensorShapeProto.Dim.size)
  return size_;
}
inline void TensorShapeProto_Dim::set_size(::google::protobuf::int64 value) {
  
  size_ = value;
  // @@protoc_insertion_point(field_set:nnfusion.serialize.TensorShapeProto.Dim.size)
}

// string name = 2;
inline void TensorShapeProto_Dim::clear_name() {
  name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline const ::std::string& TensorShapeProto_Dim::name() const {
  // @@protoc_insertion_point(field_get:nnfusion.serialize.TensorShapeProto.Dim.name)
  return name_.Get();
}
inline void TensorShapeProto_Dim::set_name(const ::std::string& value) {
  
  name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set:nnfusion.serialize.TensorShapeProto.Dim.name)
}
#if LANG_CXX11
inline void TensorShapeProto_Dim::set_name(::std::string&& value) {
  
  name_.Set(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::move(value), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_rvalue:nnfusion.serialize.TensorShapeProto.Dim.name)
}
#endif
inline void TensorShapeProto_Dim::set_name(const char* value) {
  GOOGLE_DCHECK(value != NULL);
  
  name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value),
              GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_char:nnfusion.serialize.TensorShapeProto.Dim.name)
}
inline void TensorShapeProto_Dim::set_name(const char* value,
    size_t size) {
  
  name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(
      reinterpret_cast<const char*>(value), size), GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_pointer:nnfusion.serialize.TensorShapeProto.Dim.name)
}
inline ::std::string* TensorShapeProto_Dim::mutable_name() {
  
  // @@protoc_insertion_point(field_mutable:nnfusion.serialize.TensorShapeProto.Dim.name)
  return name_.Mutable(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline ::std::string* TensorShapeProto_Dim::release_name() {
  // @@protoc_insertion_point(field_release:nnfusion.serialize.TensorShapeProto.Dim.name)
  
  return name_.Release(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
}
inline void TensorShapeProto_Dim::set_allocated_name(::std::string* name) {
  if (name != NULL) {
    
  } else {
    
  }
  name_.SetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), name,
      GetArenaNoVirtual());
  // @@protoc_insertion_point(field_set_allocated:nnfusion.serialize.TensorShapeProto.Dim.name)
}
inline ::std::string* TensorShapeProto_Dim::unsafe_arena_release_name() {
  // @@protoc_insertion_point(field_unsafe_arena_release:nnfusion.serialize.TensorShapeProto.Dim.name)
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  
  return name_.UnsafeArenaRelease(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      GetArenaNoVirtual());
}
inline void TensorShapeProto_Dim::unsafe_arena_set_allocated_name(
    ::std::string* name) {
  GOOGLE_DCHECK(GetArenaNoVirtual() != NULL);
  if (name != NULL) {
    
  } else {
    
  }
  name_.UnsafeArenaSetAllocated(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      name, GetArenaNoVirtual());
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:nnfusion.serialize.TensorShapeProto.Dim.name)
}

// -------------------------------------------------------------------

// TensorShapeProto

// repeated .nnfusion.serialize.TensorShapeProto.Dim dim = 2;
inline int TensorShapeProto::dim_size() const {
  return dim_.size();
}
inline void TensorShapeProto::clear_dim() {
  dim_.Clear();
}
inline ::nnfusion::serialize::TensorShapeProto_Dim* TensorShapeProto::mutable_dim(int index) {
  // @@protoc_insertion_point(field_mutable:nnfusion.serialize.TensorShapeProto.dim)
  return dim_.Mutable(index);
}
inline ::google::protobuf::RepeatedPtrField< ::nnfusion::serialize::TensorShapeProto_Dim >*
TensorShapeProto::mutable_dim() {
  // @@protoc_insertion_point(field_mutable_list:nnfusion.serialize.TensorShapeProto.dim)
  return &dim_;
}
inline const ::nnfusion::serialize::TensorShapeProto_Dim& TensorShapeProto::dim(int index) const {
  // @@protoc_insertion_point(field_get:nnfusion.serialize.TensorShapeProto.dim)
  return dim_.Get(index);
}
inline ::nnfusion::serialize::TensorShapeProto_Dim* TensorShapeProto::add_dim() {
  // @@protoc_insertion_point(field_add:nnfusion.serialize.TensorShapeProto.dim)
  return dim_.Add();
}
inline const ::google::protobuf::RepeatedPtrField< ::nnfusion::serialize::TensorShapeProto_Dim >&
TensorShapeProto::dim() const {
  // @@protoc_insertion_point(field_list:nnfusion.serialize.TensorShapeProto.dim)
  return dim_;
}

// bool unknown_rank = 3;
inline void TensorShapeProto::clear_unknown_rank() {
  unknown_rank_ = false;
}
inline bool TensorShapeProto::unknown_rank() const {
  // @@protoc_insertion_point(field_get:nnfusion.serialize.TensorShapeProto.unknown_rank)
  return unknown_rank_;
}
inline void TensorShapeProto::set_unknown_rank(bool value) {
  
  unknown_rank_ = value;
  // @@protoc_insertion_point(field_set:nnfusion.serialize.TensorShapeProto.unknown_rank)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace serialize
}  // namespace nnfusion

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_INCLUDED_nnf_5ftensor_5fshape_2eproto
