// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: nnf_node_def.proto

#include "nnf_node_def.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace protobuf_nnf_5fattr_5fvalue_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_nnf_5fattr_5fvalue_2eproto ::google::protobuf::internal::SCCInfo<2> scc_info_AttrValue;
}  // namespace protobuf_nnf_5fattr_5fvalue_2eproto
namespace protobuf_nnf_5fnode_5fdef_2eproto {
extern PROTOBUF_INTERNAL_EXPORT_protobuf_nnf_5fnode_5fdef_2eproto ::google::protobuf::internal::SCCInfo<1> scc_info_NodeDef_AttrEntry_DoNotUse;
}  // namespace protobuf_nnf_5fnode_5fdef_2eproto
namespace nnfusion {
namespace serialize {
class NodeDef_AttrEntry_DoNotUseDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<NodeDef_AttrEntry_DoNotUse>
      _instance;
} _NodeDef_AttrEntry_DoNotUse_default_instance_;
class NodeDefDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<NodeDef>
      _instance;
} _NodeDef_default_instance_;
}  // namespace serialize
}  // namespace nnfusion
namespace protobuf_nnf_5fnode_5fdef_2eproto {
static void InitDefaultsNodeDef_AttrEntry_DoNotUse() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::nnfusion::serialize::_NodeDef_AttrEntry_DoNotUse_default_instance_;
    new (ptr) ::nnfusion::serialize::NodeDef_AttrEntry_DoNotUse();
  }
  ::nnfusion::serialize::NodeDef_AttrEntry_DoNotUse::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<1> scc_info_NodeDef_AttrEntry_DoNotUse =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsNodeDef_AttrEntry_DoNotUse}, {
      &protobuf_nnf_5fattr_5fvalue_2eproto::scc_info_AttrValue.base,}};

static void InitDefaultsNodeDef() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::nnfusion::serialize::_NodeDef_default_instance_;
    new (ptr) ::nnfusion::serialize::NodeDef();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::nnfusion::serialize::NodeDef::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<1> scc_info_NodeDef =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsNodeDef}, {
      &protobuf_nnf_5fnode_5fdef_2eproto::scc_info_NodeDef_AttrEntry_DoNotUse.base,}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_NodeDef_AttrEntry_DoNotUse.base);
  ::google::protobuf::internal::InitSCC(&scc_info_NodeDef.base);
}

::google::protobuf::Metadata file_level_metadata[2];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::nnfusion::serialize::NodeDef_AttrEntry_DoNotUse, _has_bits_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::nnfusion::serialize::NodeDef_AttrEntry_DoNotUse, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::nnfusion::serialize::NodeDef_AttrEntry_DoNotUse, key_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::nnfusion::serialize::NodeDef_AttrEntry_DoNotUse, value_),
  0,
  1,
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::nnfusion::serialize::NodeDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::nnfusion::serialize::NodeDef, name_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::nnfusion::serialize::NodeDef, op_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::nnfusion::serialize::NodeDef, input_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::nnfusion::serialize::NodeDef, attr_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, sizeof(::nnfusion::serialize::NodeDef_AttrEntry_DoNotUse)},
  { 9, -1, sizeof(::nnfusion::serialize::NodeDef)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::nnfusion::serialize::_NodeDef_AttrEntry_DoNotUse_default_instance_),
  reinterpret_cast<const ::google::protobuf::Message*>(&::nnfusion::serialize::_NodeDef_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "nnf_node_def.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 2);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\022nnf_node_def.proto\022\022nnfusion.serialize"
      "\032\024nnf_attr_value.proto\"\263\001\n\007NodeDef\022\014\n\004na"
      "me\030\001 \001(\t\022\n\n\002op\030\002 \001(\t\022\r\n\005input\030\003 \003(\t\0223\n\004a"
      "ttr\030\005 \003(\0132%.nnfusion.serialize.NodeDef.A"
      "ttrEntry\032J\n\tAttrEntry\022\013\n\003key\030\001 \001(\t\022,\n\005va"
      "lue\030\002 \001(\0132\035.nnfusion.serialize.AttrValue"
      ":\0028\001B\003\370\001\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 257);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "nnf_node_def.proto", &protobuf_RegisterTypes);
  ::protobuf_nnf_5fattr_5fvalue_2eproto::AddDescriptors();
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_nnf_5fnode_5fdef_2eproto
namespace nnfusion {
namespace serialize {

// ===================================================================

NodeDef_AttrEntry_DoNotUse::NodeDef_AttrEntry_DoNotUse() {}
NodeDef_AttrEntry_DoNotUse::NodeDef_AttrEntry_DoNotUse(::google::protobuf::Arena* arena) : SuperType(arena) {}
void NodeDef_AttrEntry_DoNotUse::MergeFrom(const NodeDef_AttrEntry_DoNotUse& other) {
  MergeFromInternal(other);
}
::google::protobuf::Metadata NodeDef_AttrEntry_DoNotUse::GetMetadata() const {
  ::protobuf_nnf_5fnode_5fdef_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_nnf_5fnode_5fdef_2eproto::file_level_metadata[0];
}
void NodeDef_AttrEntry_DoNotUse::MergeFrom(
    const ::google::protobuf::Message& other) {
  ::google::protobuf::Message::MergeFrom(other);
}


// ===================================================================

void NodeDef::InitAsDefaultInstance() {
}
void NodeDef::clear_attr() {
  attr_.Clear();
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int NodeDef::kNameFieldNumber;
const int NodeDef::kOpFieldNumber;
const int NodeDef::kInputFieldNumber;
const int NodeDef::kAttrFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

NodeDef::NodeDef()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_nnf_5fnode_5fdef_2eproto::scc_info_NodeDef.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:nnfusion.serialize.NodeDef)
}
NodeDef::NodeDef(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena),
  input_(arena),
  attr_(arena) {
  ::google::protobuf::internal::InitSCC(&protobuf_nnf_5fnode_5fdef_2eproto::scc_info_NodeDef.base);
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:nnfusion.serialize.NodeDef)
}
NodeDef::NodeDef(const NodeDef& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      input_(from.input_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  attr_.MergeFrom(from.attr_);
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.name().size() > 0) {
    name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.name(),
      GetArenaNoVirtual());
  }
  op_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.op().size() > 0) {
    op_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.op(),
      GetArenaNoVirtual());
  }
  // @@protoc_insertion_point(copy_constructor:nnfusion.serialize.NodeDef)
}

void NodeDef::SharedCtor() {
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  op_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

NodeDef::~NodeDef() {
  // @@protoc_insertion_point(destructor:nnfusion.serialize.NodeDef)
  SharedDtor();
}

void NodeDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
  name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  op_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void NodeDef::ArenaDtor(void* object) {
  NodeDef* _this = reinterpret_cast< NodeDef* >(object);
  (void)_this;
}
void NodeDef::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void NodeDef::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* NodeDef::descriptor() {
  ::protobuf_nnf_5fnode_5fdef_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_nnf_5fnode_5fdef_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const NodeDef& NodeDef::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_nnf_5fnode_5fdef_2eproto::scc_info_NodeDef.base);
  return *internal_default_instance();
}


void NodeDef::Clear() {
// @@protoc_insertion_point(message_clear_start:nnfusion.serialize.NodeDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  input_.Clear();
  attr_.Clear();
  name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  op_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  _internal_metadata_.Clear();
}

bool NodeDef::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:nnfusion.serialize.NodeDef)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string name = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->name().data(), static_cast<int>(this->name().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "nnfusion.serialize.NodeDef.name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string op = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_op()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->op().data(), static_cast<int>(this->op().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "nnfusion.serialize.NodeDef.op"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated string input = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(26u /* 26 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->add_input()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->input(this->input_size() - 1).data(),
            static_cast<int>(this->input(this->input_size() - 1).length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "nnfusion.serialize.NodeDef.input"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // map<string, .nnfusion.serialize.AttrValue> attr = 5;
      case 5: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(42u /* 42 & 0xFF */)) {
          NodeDef_AttrEntry_DoNotUse::Parser< ::google::protobuf::internal::MapField<
              NodeDef_AttrEntry_DoNotUse,
              ::std::string, ::nnfusion::serialize::AttrValue,
              ::google::protobuf::internal::WireFormatLite::TYPE_STRING,
              ::google::protobuf::internal::WireFormatLite::TYPE_MESSAGE,
              0 >,
            ::google::protobuf::Map< ::std::string, ::nnfusion::serialize::AttrValue > > parser(&attr_);
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
              input, &parser));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            parser.key().data(), static_cast<int>(parser.key().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "nnfusion.serialize.NodeDef.AttrEntry.key"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:nnfusion.serialize.NodeDef)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:nnfusion.serialize.NodeDef)
  return false;
#undef DO_
}

void NodeDef::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:nnfusion.serialize.NodeDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string name = 1;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "nnfusion.serialize.NodeDef.name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->name(), output);
  }

  // string op = 2;
  if (this->op().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->op().data(), static_cast<int>(this->op().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "nnfusion.serialize.NodeDef.op");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->op(), output);
  }

  // repeated string input = 3;
  for (int i = 0, n = this->input_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->input(i).data(), static_cast<int>(this->input(i).length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "nnfusion.serialize.NodeDef.input");
    ::google::protobuf::internal::WireFormatLite::WriteString(
      3, this->input(i), output);
  }

  // map<string, .nnfusion.serialize.AttrValue> attr = 5;
  if (!this->attr().empty()) {
    typedef ::google::protobuf::Map< ::std::string, ::nnfusion::serialize::AttrValue >::const_pointer
        ConstPtr;
    typedef ConstPtr SortItem;
    typedef ::google::protobuf::internal::CompareByDerefFirst<SortItem> Less;
    struct Utf8Check {
      static void Check(ConstPtr p) {
        ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
          p->first.data(), static_cast<int>(p->first.length()),
          ::google::protobuf::internal::WireFormatLite::SERIALIZE,
          "nnfusion.serialize.NodeDef.AttrEntry.key");
      }
    };

    if (output->IsSerializationDeterministic() &&
        this->attr().size() > 1) {
      ::std::unique_ptr<SortItem[]> items(
          new SortItem[this->attr().size()]);
      typedef ::google::protobuf::Map< ::std::string, ::nnfusion::serialize::AttrValue >::size_type size_type;
      size_type n = 0;
      for (::google::protobuf::Map< ::std::string, ::nnfusion::serialize::AttrValue >::const_iterator
          it = this->attr().begin();
          it != this->attr().end(); ++it, ++n) {
        items[static_cast<ptrdiff_t>(n)] = SortItem(&*it);
      }
      ::std::sort(&items[0], &items[static_cast<ptrdiff_t>(n)], Less());
      ::std::unique_ptr<NodeDef_AttrEntry_DoNotUse> entry;
      for (size_type i = 0; i < n; i++) {
        entry.reset(attr_.NewEntryWrapper(
            items[static_cast<ptrdiff_t>(i)]->first, items[static_cast<ptrdiff_t>(i)]->second));
        ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
            5, *entry, output);
        if (entry->GetArena() != NULL) {
          entry.release();
        }
        Utf8Check::Check(items[static_cast<ptrdiff_t>(i)]);
      }
    } else {
      ::std::unique_ptr<NodeDef_AttrEntry_DoNotUse> entry;
      for (::google::protobuf::Map< ::std::string, ::nnfusion::serialize::AttrValue >::const_iterator
          it = this->attr().begin();
          it != this->attr().end(); ++it) {
        entry.reset(attr_.NewEntryWrapper(
            it->first, it->second));
        ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
            5, *entry, output);
        if (entry->GetArena() != NULL) {
          entry.release();
        }
        Utf8Check::Check(&*it);
      }
    }
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:nnfusion.serialize.NodeDef)
}

::google::protobuf::uint8* NodeDef::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:nnfusion.serialize.NodeDef)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string name = 1;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "nnfusion.serialize.NodeDef.name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->name(), target);
  }

  // string op = 2;
  if (this->op().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->op().data(), static_cast<int>(this->op().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "nnfusion.serialize.NodeDef.op");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        2, this->op(), target);
  }

  // repeated string input = 3;
  for (int i = 0, n = this->input_size(); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->input(i).data(), static_cast<int>(this->input(i).length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "nnfusion.serialize.NodeDef.input");
    target = ::google::protobuf::internal::WireFormatLite::
      WriteStringToArray(3, this->input(i), target);
  }

  // map<string, .nnfusion.serialize.AttrValue> attr = 5;
  if (!this->attr().empty()) {
    typedef ::google::protobuf::Map< ::std::string, ::nnfusion::serialize::AttrValue >::const_pointer
        ConstPtr;
    typedef ConstPtr SortItem;
    typedef ::google::protobuf::internal::CompareByDerefFirst<SortItem> Less;
    struct Utf8Check {
      static void Check(ConstPtr p) {
        ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
          p->first.data(), static_cast<int>(p->first.length()),
          ::google::protobuf::internal::WireFormatLite::SERIALIZE,
          "nnfusion.serialize.NodeDef.AttrEntry.key");
      }
    };

    if (deterministic &&
        this->attr().size() > 1) {
      ::std::unique_ptr<SortItem[]> items(
          new SortItem[this->attr().size()]);
      typedef ::google::protobuf::Map< ::std::string, ::nnfusion::serialize::AttrValue >::size_type size_type;
      size_type n = 0;
      for (::google::protobuf::Map< ::std::string, ::nnfusion::serialize::AttrValue >::const_iterator
          it = this->attr().begin();
          it != this->attr().end(); ++it, ++n) {
        items[static_cast<ptrdiff_t>(n)] = SortItem(&*it);
      }
      ::std::sort(&items[0], &items[static_cast<ptrdiff_t>(n)], Less());
      ::std::unique_ptr<NodeDef_AttrEntry_DoNotUse> entry;
      for (size_type i = 0; i < n; i++) {
        entry.reset(attr_.NewEntryWrapper(
            items[static_cast<ptrdiff_t>(i)]->first, items[static_cast<ptrdiff_t>(i)]->second));
        target = ::google::protobuf::internal::WireFormatLite::
                   InternalWriteMessageNoVirtualToArray(
                       5, *entry, deterministic, target);
;
        if (entry->GetArena() != NULL) {
          entry.release();
        }
        Utf8Check::Check(items[static_cast<ptrdiff_t>(i)]);
      }
    } else {
      ::std::unique_ptr<NodeDef_AttrEntry_DoNotUse> entry;
      for (::google::protobuf::Map< ::std::string, ::nnfusion::serialize::AttrValue >::const_iterator
          it = this->attr().begin();
          it != this->attr().end(); ++it) {
        entry.reset(attr_.NewEntryWrapper(
            it->first, it->second));
        target = ::google::protobuf::internal::WireFormatLite::
                   InternalWriteMessageNoVirtualToArray(
                       5, *entry, deterministic, target);
;
        if (entry->GetArena() != NULL) {
          entry.release();
        }
        Utf8Check::Check(&*it);
      }
    }
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:nnfusion.serialize.NodeDef)
  return target;
}

size_t NodeDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:nnfusion.serialize.NodeDef)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated string input = 3;
  total_size += 1 *
      ::google::protobuf::internal::FromIntSize(this->input_size());
  for (int i = 0, n = this->input_size(); i < n; i++) {
    total_size += ::google::protobuf::internal::WireFormatLite::StringSize(
      this->input(i));
  }

  // map<string, .nnfusion.serialize.AttrValue> attr = 5;
  total_size += 1 *
      ::google::protobuf::internal::FromIntSize(this->attr_size());
  {
    ::std::unique_ptr<NodeDef_AttrEntry_DoNotUse> entry;
    for (::google::protobuf::Map< ::std::string, ::nnfusion::serialize::AttrValue >::const_iterator
        it = this->attr().begin();
        it != this->attr().end(); ++it) {
      if (entry.get() != NULL && entry->GetArena() != NULL) {
        entry.release();
      }
      entry.reset(attr_.NewEntryWrapper(it->first, it->second));
      total_size += ::google::protobuf::internal::WireFormatLite::
          MessageSizeNoVirtual(*entry);
    }
    if (entry.get() != NULL && entry->GetArena() != NULL) {
      entry.release();
    }
  }

  // string name = 1;
  if (this->name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->name());
  }

  // string op = 2;
  if (this->op().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->op());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void NodeDef::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:nnfusion.serialize.NodeDef)
  GOOGLE_DCHECK_NE(&from, this);
  const NodeDef* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const NodeDef>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:nnfusion.serialize.NodeDef)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:nnfusion.serialize.NodeDef)
    MergeFrom(*source);
  }
}

void NodeDef::MergeFrom(const NodeDef& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:nnfusion.serialize.NodeDef)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  input_.MergeFrom(from.input_);
  attr_.MergeFrom(from.attr_);
  if (from.name().size() > 0) {
    set_name(from.name());
  }
  if (from.op().size() > 0) {
    set_op(from.op());
  }
}

void NodeDef::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:nnfusion.serialize.NodeDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void NodeDef::CopyFrom(const NodeDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:nnfusion.serialize.NodeDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool NodeDef::IsInitialized() const {
  return true;
}

void NodeDef::Swap(NodeDef* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    NodeDef* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void NodeDef::UnsafeArenaSwap(NodeDef* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void NodeDef::InternalSwap(NodeDef* other) {
  using std::swap;
  input_.InternalSwap(CastToBase(&other->input_));
  attr_.Swap(&other->attr_);
  name_.Swap(&other->name_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  op_.Swap(&other->op_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata NodeDef::GetMetadata() const {
  protobuf_nnf_5fnode_5fdef_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_nnf_5fnode_5fdef_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace serialize
}  // namespace nnfusion
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::nnfusion::serialize::NodeDef_AttrEntry_DoNotUse* Arena::CreateMaybeMessage< ::nnfusion::serialize::NodeDef_AttrEntry_DoNotUse >(Arena* arena) {
  return Arena::CreateMessageInternal< ::nnfusion::serialize::NodeDef_AttrEntry_DoNotUse >(arena);
}
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::nnfusion::serialize::NodeDef* Arena::CreateMaybeMessage< ::nnfusion::serialize::NodeDef >(Arena* arena) {
  return Arena::CreateMessageInternal< ::nnfusion::serialize::NodeDef >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
