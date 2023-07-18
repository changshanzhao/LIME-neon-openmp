add_library( feature SHARED lime_neon_OpenMP.cpp)

target_link_libraries( feature
    ${THIRD_PARTY_LIBS}
)
