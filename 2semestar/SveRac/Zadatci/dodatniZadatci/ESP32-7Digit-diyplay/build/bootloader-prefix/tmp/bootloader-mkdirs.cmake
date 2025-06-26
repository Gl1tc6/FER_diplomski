# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/FER/2_Diplomski/2_Semestar/Sveprisutno_racunarstvo/ESP_config/v5.4/esp-idf/components/bootloader/subproject")
  file(MAKE_DIRECTORY "D:/FER/2_Diplomski/2_Semestar/Sveprisutno_racunarstvo/ESP_config/v5.4/esp-idf/components/bootloader/subproject")
endif()
file(MAKE_DIRECTORY
  "D:/FER/2_Diplomski/2_Semestar/Sveprisutno_racunarstvo/ZI_zadaci/ESP32-napon-mreze/build/bootloader"
  "D:/FER/2_Diplomski/2_Semestar/Sveprisutno_racunarstvo/ZI_zadaci/ESP32-napon-mreze/build/bootloader-prefix"
  "D:/FER/2_Diplomski/2_Semestar/Sveprisutno_racunarstvo/ZI_zadaci/ESP32-napon-mreze/build/bootloader-prefix/tmp"
  "D:/FER/2_Diplomski/2_Semestar/Sveprisutno_racunarstvo/ZI_zadaci/ESP32-napon-mreze/build/bootloader-prefix/src/bootloader-stamp"
  "D:/FER/2_Diplomski/2_Semestar/Sveprisutno_racunarstvo/ZI_zadaci/ESP32-napon-mreze/build/bootloader-prefix/src"
  "D:/FER/2_Diplomski/2_Semestar/Sveprisutno_racunarstvo/ZI_zadaci/ESP32-napon-mreze/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/FER/2_Diplomski/2_Semestar/Sveprisutno_racunarstvo/ZI_zadaci/ESP32-napon-mreze/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/FER/2_Diplomski/2_Semestar/Sveprisutno_racunarstvo/ZI_zadaci/ESP32-napon-mreze/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
