import jpype
import jpype.imports
import os

try:
    if not jpype.isJVMStarted():
        mage_jar_path = "./mage/Mage/target/mage.jar"
        jpype.startJVM(
            jpype.getDefaultJVMPath(),
            f"-Djava.class.path={mage_jar_path}",
            convertStrings=False
        )
    
    # Get the mage package
    mage = jpype.JPackage('mage')
    
    print("Exploring MAGE package structure...")
    
    # Try to list sub-packages under mage
    print("\nTrying to list mage sub-packages:")
    try:
        # This might not work for all Java packages, but let's try
        mage_pkg = jpype.JClass('java.lang.Package').getPackage('mage')
        if mage_pkg:
            print(f"Found mage package: {mage_pkg}")
    except:
        print("Could not get package info directly")
    
    # Try to discover classes by attempting to access common patterns
    common_card_classes = [
        'mage.cards.repository.CardRepository',
        'mage.cards.Cards',
        'mage.cards.repository.CardInfo',
        'mage.sets.Sets',
        'mage.sets.CardSets',
        'mage.cards.decks.Deck',
        'mage.game.Game',
        'mage.players.Player'
    ]
    
    print("\nTesting common MAGE class patterns:")
    for class_name in common_card_classes:
        try:
            cls = jpype.JClass(class_name)
            print(f"✓ Found: {class_name}")
            # Try to get instance if it's a singleton
            if hasattr(cls, 'instance'):
                instance = cls.instance
                print(f"  Instance available: {instance}")
        except:
            print(f"✗ Not found: {class_name}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if jpype.isJVMStarted():
        jpype.shutdownJVM()